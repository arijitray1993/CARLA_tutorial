# CREATE A DATASET WITH VARYING WEATHER AND NUMBER OF VEHICLES. 

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import queue
import cv2
import skimage.measure as measure
import pycocotools
import json
import tqdm
import numpy as np
from detectron2.structures import BoxMode

import pickle as pkl

import pdb


def create_dataset(args, client, global_count):
    print("loading configs...")
    if not os.path.exists('images'):
        os.makedirs('images')

    if not os.path.exists('images_semseg'):
        os.makedirs('images_semseg')

    if not os.path.exists('images_depth'):
        os.makedirs('images_depth')

    #generate a map, drive around, get images
    world = client.load_world(args['town_choice'])
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.03 #must be less than 0.1, or else physics will be noisy
    #must use fixed delta seconds and synchronous mode for python api controlled sim, or else 
    #python may not be able to keep up with simulator. 
    settings.synchronous_mode = True 
    world.apply_settings(settings)

    weather = carla.WeatherParameters(
        cloudiness=args['cloud'],
        precipitation=args['rain'],
        sun_altitude_angle=args['sun_angle'])
    #or use precomputed weathers
    #weather = carla.WeatherParameters.WetCloudySunset

    world.set_weather(weather)

    # let's add stuff to the world
    actor_list = []
    blueprint_library = world.get_blueprint_library()
    bp = random.choice(blueprint_library.filter('vehicle')) # lets choose a vehicle at random

    # lets choose a random spawn point
    transform = random.choice(world.get_map().get_spawn_points()) 

    #spawn a vehicle
    vehicle = world.spawn_actor(bp, transform)
    actor_list.append(vehicle)

    vehicle.set_autopilot(True)


    #lets create waypoints for driving the vehicle around automatically
    m= world.get_map()
    waypoint = m.get_waypoint(transform.location)

    #lets add more vehicles
    for _ in range(0, args['num_vehicles']):
        transform = random.choice(m.get_spawn_points())

        bp = random.choice(blueprint_library.filter('vehicle'))

        # This time we are using try_spawn_actor. If the spot is already
        # occupied by another object, the function will return None.
        npc = world.try_spawn_actor(bp, transform)
        if npc is not None:
            print(npc)
            npc.set_autopilot(True)
            actor_list.append(npc)

            
    #example for getting camera image
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    image_queue = queue.Queue()
    camera.listen(image_queue.put)
    actor_list.append(camera)

    #example for getting depth camera image
    camera_depth = blueprint_library.find('sensor.camera.depth')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera_d = world.spawn_actor(camera_depth, camera_transform, attach_to=vehicle)
    image_queue_depth = queue.Queue()
    camera_d.listen(image_queue_depth.put)
    actor_list.append(camera_d)

    #example for getting semantic segmentation camera image
    camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera_seg = world.spawn_actor(camera_semseg, camera_transform, attach_to=vehicle)
    image_queue_seg = queue.Queue()
    camera_seg.listen(image_queue_seg.put)
    actor_list.append(camera_seg)

    #try:
    #init the COCO annotation dict to be populated. 
    dataset_dicts = []

    print("creating dataset: "+args['name'])

    for i in tqdm.tqdm(range(args['num_frames'])):
        #step
        world.tick()
        
        #rgb camera
        image = image_queue.get()
        
        #semantic segmentation camera
        image_seg  = image_queue_seg.get()
        #image_seg.convert(carla.ColorConverter.CityScapesPalette)

        #depth camera
        image_depth = image_queue_depth.get()
        #image_depth.convert(carla.ColorConverter.LogarithmicDepth)

        
        if i%args['sample_every']==0:
            image.save_to_disk("images/%s_%06d.png" %(args['name'], image.frame))
            image_seg.save_to_disk("images_semseg/%s_%06d.png" %(args['name'], image.frame), carla.ColorConverter.CityScapesPalette)
            image_depth.save_to_disk("images_depth/%s_%06d.png" %(args['name'], image.frame), carla.ColorConverter.LogarithmicDepth)

            ## COCO format stuff, each image needs to have these keys
            height, width = cv2.imread("images/%s_%06d.png" %(args['name'], image.frame)).shape[:2]
            record = {}
            record['file_name'] = "images/%s_%06d.png" %(args['name'], image.frame)
            global_count+=1
            record['image_id'] = global_count
            record['height'] = height
            record['width'] = width


            ## compute bboxes from semantic segmentation image
            img_semseg_bgr = cv2.imread("images_semseg/%s_%06d.png" %(args['name'], image.frame))
            img_semseg_bgr = cv2.cvtColor(img_semseg_bgr, cv2.COLOR_BGRA2BGR)
            img_semseg_hsv = cv2.cvtColor(img_semseg_bgr, cv2.COLOR_BGR2HSV) # color wise segmentation is better in hsv space

            #bgr value exmaples of few objects: full list at https://carla.readthedocs.io/en/0.9.9/ref_sensors/ 
            object_list = dict()
            object_list['building'] = np.uint8([[[70, 70, 70]]])        
            object_list['pedestrian'] = np.uint8([[[220, 20, 60]]])
            object_list['vegetation'] = np.uint8([[[107, 142, 35]]])
            object_list['car'] = np.uint8([[[ 0, 0, 142]]])
            object_list['fence'] = np.uint8([[[ 190, 153, 153]]])
            object_list['traffic_sign'] = np.uint8([[[220, 220, 0]]])
            object_list['pole'] = np.uint8([[[153, 153, 153]]])
            object_list['wall'] = np.uint8([[[102, 102, 156]]])
            
            objects = []
            obj_id = 0
            obj2id = dict()
            for obj in object_list:
                mask = get_mask(img_semseg_hsv, object_list[obj])
                bboxes = get_bbox_from_mask(mask) # minr, minc, maxr, maxc
                
                #let's visualize car bboxes
                #if obj=='car':
                #    ax4.imshow(mask)
                #    for bbox in bboxes:
                #        minr, minc, maxr, maxc = bbox
                #        cv2.rectangle(img_semseg_bgr, (minc,minr), (maxc, maxr), (255,255,255), 6)
                
                ## Now let's put these bboxes and semantic segmentation masks into COCO format. 
                for bbox in bboxes:
                    minr, minc, maxr, maxc = bbox
                    #obj_mask = np.copy(mask)
                    #obj_mask[:minr] = 0
                    #obj_mask[:, :minc] = 0
                    #obj_mask[maxr+1:] = 0
                    #obj_mask[:, maxc+1:] = 0
                    
                    #coco_rle_mask = pycocotools.mask.encode(np.array(obj_mask, order="F")) # make sure to set cfg.INPUT.MASK_FORMAT = bitmask when loading cfg for detectron2. 

                    obj_ann = {
                        'bbox': [minc, minr, maxc, maxr],
                        #'bbox_mode': BoxMode.XYXY_ABS,
                        #'segmentation': coco_rle_mask,
                        #'category_id': obj_id,
                        #'mask_segmentation': obj_mask,
                        'obj_name': obj
                    }
                    #saving mask makes the file huge, so, it's best to compute it while loading. 

                    objects.append(obj_ann)

                obj_id+=1
                obj2id[obj] = obj_id

            record['annotations'] = objects
            dataset_dicts.append(record)
        

        #drive vehicle to next waypoint on map
        waypoint = random.choice(waypoint.next(1.5))
        vehicle.set_transform(waypoint.transform)
    #except:
    #    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
    camera.destroy()
    camera_seg.destroy()
    camera_d.destroy()

    client.apply_batch([carla.command.DestroyActor(x) for x in actor_list]) # must destroy stuff to not having them linger around for next time

    #save dataset
    return dataset_dicts, global_count

    

def get_mask(seg_im, rgb_value):
    # rgb_value should be somethiing like np.uint8([[[70, 70, 70]]])
    # seg_im should be in HSV
    
    hsv_value = cv2.cvtColor(rgb_value, cv2.COLOR_RGB2HSV)
    
    hsv_low = np.array([[[hsv_value[0][0][0]-5, hsv_value[0][0][1], hsv_value[0][0][2]-5]]])
    hsv_high = np.array([[[hsv_value[0][0][0]+5, hsv_value[0][0][1], hsv_value[0][0][2]+5]]])
    
    mask = cv2.inRange(seg_im, hsv_low, hsv_high)
    return mask

def get_bbox_from_mask(mask):
    label_mask = measure.label(mask)
    props = measure.regionprops(label_mask)
    
    return [prop.bbox for prop in props]


if __name__=="__main__":

    client = carla.Client('localhost', 2000)
    client.set_timeout(20.0)

    all_dataset_dicts = []

    global_count=0

    #number of image frames saved would be num_frames/sample_every
    dataset_args={
        'name': 'sunny_noon_busy',
        'town_choice': 'Town03',
        'cloud': 20,
        'rain': 20,
        'sun_angle': 100,
        'num_vehicles': 200,
        'num_frames': 10000,
        'sample_every': 5
    }

    data_dicts, global_count = create_dataset(dataset_args, client, global_count)
    all_dataset_dicts.extend(data_dicts)

    #pdb.set_trace()
    with open("COCO_annotation.pkl", "wb") as f:
        pkl.dump(all_dataset_dicts, f)


    #sys.exit(0)
    
    dataset_args={
        'name': 'cloudy_morning_busy',
        'town_choice': 'Town03',
        'cloud': 80,
        'rain': 20,
        'sun_angle': 40,
        'num_vehicles': 200,
        'num_frames': 10000,
        'sample_every': 5
    }

    data_dicts, global_count = create_dataset(dataset_args, client, global_count)
    all_dataset_dicts.extend(data_dicts)

    with open("COCO_annotation.pkl", "wb") as f:
        pkl.dump(all_dataset_dicts, f)

    dataset_args={
        'name': 'raining_evening_busy',
        'town_choice': 'Town03',
        'cloud': 80,
        'rain': 80,
        'sun_angle': 40,
        'num_vehicles': 200,
        'num_frames': 10000,
        'sample_every': 5
    }

    data_dicts, global_count = create_dataset(dataset_args, client, global_count)
    all_dataset_dicts.extend(data_dicts)

    with open("COCO_annotation.pkl", "wb") as f:
        pkl.dump(all_dataset_dicts, f)

    dataset_args={
        'name': 'sunny_evening_busy',
        'town_choice': 'Town03',
        'cloud': 10,
        'rain': 0,
        'sun_angle': 130,
        'num_vehicles': 200,
        'num_frames': 10000,
        'sample_every': 5
    }

    data_dicts, global_count = create_dataset(dataset_args, client, global_count)
    all_dataset_dicts.extend(data_dicts)

    with open("COCO_annotation.pkl", "wb") as f:
        pkl.dump(all_dataset_dicts, f)

    dataset_args={
        'name': 'night_busy',
        'town_choice': 'Town03',
        'cloud': 10,
        'rain': 0,
        'sun_angle': 190,
        'num_vehicles': 200,
        'num_frames': 10000,
        'sample_every': 5
    }

    data_dicts, global_count = create_dataset(dataset_args, client, global_count)
    all_dataset_dicts.extend(data_dicts)

    
    with open("COCO_annotation.pkl", "wb") as f:
        pkl.dump(all_dataset_dicts, f)

    



