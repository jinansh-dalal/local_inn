import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from pathlib import Path

# --- CONFIGURATION ---
ROSBAG_PATH = Path('./v4.db3')  # <--- CHANGE THIS to your actual bag path
OUTPUT_FILENAME = 'my_rosbag_data.npz'
SCAN_TOPIC = '/autodrive/roboracer_1/lidar'              # Check your topic names!
ODOM_TOPIC = '/tf'              # Check your topic names!

typestore = get_typestore(Stores.ROS2_HUMBLE)

# LiDAR Config (Critical for your 270 deg sensor)
LIDAR_RAYS = 270 

def process_bag():
    scans = []
    poses = []
    
    print(f"Reading {ROSBAG_PATH}...")
    
    # We'll use a simple matching strategy: 
    # For every scan, find the most recent odom pose.
    last_pose = None
    
    with AnyReader([ROSBAG_PATH], default_typestore=typestore) as reader:
        # Get connection info for the topics we care about
        connections = [x for x in reader.connections if x.topic in [SCAN_TOPIC, ODOM_TOPIC]]
        
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            
            if connection.topic == ODOM_TOPIC:
                # Extract Pose (x, y) and Orientation (Quaternion -> Theta)
                px = msg.transforms[0].transform.translation.x
                py = msg.transforms[0].transform.translation.y
                
                # Quaternion to Yaw (Theta)
                qx = msg.transforms[0].transform.rotation.x
                qy = msg.transforms[0].transform.rotation.y
                qz = msg.transforms[0].transform.rotation.z
                qw = msg.transforms[0].transform.rotation.w
                
                # Yaw calculation
                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                theta = np.arctan2(siny_cosp, cosy_cosp)
                
                last_pose = np.array([px, py, theta])
                
            elif connection.topic == SCAN_TOPIC:
                if last_pose is None:
                    continue # Skip scans until we find our first pose
                
                # Extract Ranges
                # Note: We need exactly 270 rays.
                # If your sensor gives 1080 rays, we strip/downsample here.
                ranges = np.array(msg.ranges[1::4])
                
                # Handle Infinite/NaN values
                ranges[np.isinf(ranges)] = 30.0 # Max range
                ranges[np.isnan(ranges)] = 30.0
                
                scans.append(ranges)
                poses.append(last_pose)

    # Convert to Numpy Arrays
    poses = np.array(poses)
    scans = np.array(scans)
    
    print(f"Poses: {poses.shape} | Scans: {scans.shape}")
    
    # Concatenate: [Pose (3) | Scan (270)]
    # Order: [x, y, theta, range_0, ... range_269]
    data_record = np.concatenate([poses, scans], axis=1)
    
    # Save
    print(f"Saving to {OUTPUT_FILENAME}...")
    np.savez(OUTPUT_FILENAME, data_record=data_record)
    print("Done!")

if __name__ == "__main__":
    process_bag()