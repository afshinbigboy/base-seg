import argparse


def get_parser():
    
    parser = argparse.ArgumentParser(prog='BaseRetinaDNS', description='DNS has no description')
    
    parser.add_argument('--train', action='store_true', help="train mode")
    
    parser.add_argument('--test', action='store_true', help="test mode")
    
    parser.add_argument('-c', '--countinue', action='store_true', 
                        help="countinue training with loading the most recent weigths")
    
    parser.add_argument('-n', '--run-name', default='BASE_DRIVE_SEG_01', 
                        help="running name. it is like an ID for this run")
    
    parser.add_argument('--output-dir', default="/cabinet/afshin/base_seg/saved_model", 
                        help="output directory")
    
    parser.add_argument('--data-dir', default='/cabinet/dataset/Retinal/DRIVE', 
                        help="data directory")
    
    return parser

