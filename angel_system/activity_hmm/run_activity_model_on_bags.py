import os
import subprocess
import argparse
import libtmux
import time
from pathlib import Path


def run(args):
    tmux_script = 'record_ros_bag_activity_only'
    model_name = os.path.basename(args.activity_model).split('.')[0]

    server = libtmux.Server()
    env = os.environ.copy()
    
    
    for bag in args.bags:
        done = False
        print(f'Bag: {bag}')

        env['PARAM_ROS_BAG_DIR'] = f'{args.bags_dir}/{bag}'
        env['PARAM_ACTIVITY_MODEL'] = args.activity_model
        output_bag = f'{bag}_{model_name}'
        env['PARAM_ROS_BAG_OUT'] = output_bag

        start = subprocess.Popen(['tmuxinator', 'start', tmux_script],
                             env=env)
        time.sleep(1)
        try:
            session = server.sessions.get(session_name='record-ros-bag-activity-only')
            print(session)
        except:
            return
        ros2_bag = session.windows.get(window_name='ros2_bag')
        ros2_bag_pane = ros2_bag.panes[0]

        ros_bag_record = session.windows.get(window_name='ros_bag_record')
        ros_bag_record_pane = ros_bag_record.panes[0]

        # Wait until bag is done playing
        while not done:
            stdout = ros2_bag_pane.capture_pane()

            if 'done' in stdout:
                done = True
            else:
                time.sleep(60)
        
        ros_bag_record_pane.send_keys('C-c')
        time.sleep(30)

        # Wait until new bag is written
        finished_recording = False
        while not finished_recording:
            finished_recording = os.path.exists(f'{args.bags_dir}/{output_bag}/metadata.yaml')
            time.sleep(60)
        
        # Kill session
        session.kill_session()
        stop = subprocess.Popen(['tmuxinator', 'stop', tmux_script])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--activity_model",
        dest="activity_model",
        type=str,
        default='uho_checkpoint_20221022.ckpt'
    )
    parser.add_argument(
        "--split",
        dest="split",
        type=str,
        required=False,
        help="alias either by name (alex/hannah/brian) or training split (train/val/test) to grab the bag split defined by v1.3 of the dataset"
    )
    parser.add_argument(
        '--bags_dir',
        dest='bags_dir',
        type=Path,
        default='ros_bags'
    )
    parser.add_argument(
        "--bags",
        dest="bags",
        nargs='+',
        type=int,
        default=[],
        help='Bag ids'
    )

    args = parser.parse_args()

    # Add pre-defined dataset splits
    if args.split:
        if args.split == 'alex' or args.split == 'train':
            args.bags.append(range(1,23) + range(25, 41) + range(46,51))
        if args.split == 'hannah' or args.split == 'val':
            args.bags.append([23, 24, 41, 42, 43, 44, 45])
        if args.split == 'brian' or args.split ==  'test':
            args.bags.append([51, 52, 53, 54])

    # Reformat bag names
    args.bags = [f'all_activities_{bag_id}' for bag_id in args.bags]

    run(args)


    


if __name__ == '__main__':
    main()