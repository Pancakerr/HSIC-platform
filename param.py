import argparse

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--DATASET', type=str, default='IP', help="Dataset to use.(IP, UP, UH)")
    parser.add_argument('--MODEL', type=str, default='SSRNet',
                        help="Select Model. Available:\n"
                            "SSRNet"
                            "A2S2KResNet"
                            "CNN1D"
                            "FullyContNet"
                            "GhostNet"
                            "HybridSN"
                            "M3DCNN"
                            "MLWBDN"
                            "MobileNet"
                            "SSFCN"
                            "SSUN"
                            "UNet"
                            "SFormer_px"
                            "SFormer_pt"
                            "VIT"
                            )
    parser.add_argument('--MODEL_MODE', type=int, default= None,
                        help="optional for SSRNet and set 0 for other models (defaults to 0)")
    parser.add_argument('--TRAIN_RATE', type=float, default=150,
                        help="number of samples for training. (samples per class or percentage, default to 150)")
    parser.add_argument('--VAL_RATE', type=float, default=50,
                        help="number of samples for Valuating. (samples per class or percentage,default to 50)")
    parser.add_argument('--EPOCH', type=int, default= None,
                        help="epoch of training ")
    parser.add_argument('--VAL_EPOCH', type=int, default=1,
                        help="epoch interval for validation(default to 1)")
    parser.add_argument('--SAMPLE_MODE', type=str, default=None,
                        help="sample mode in PWS SLS FIS PPTR(default to PWS)")
    parser.add_argument('--PPTR_RATE', type=float, default= None,
                        help="rate of random sampling after PPTR")
    parser.add_argument('--DEVICE', type=int, default=0,
                        help="Specify CUDA device (defaults to 0, learns on GPU 0 and -1 for CPU)")
    parser.add_argument('--N_PCA', type=int, default=0,
                        help="reserved PCA components, 0 for use origin data (defaults to 0)")
    parser.add_argument('--NORM',  action='store_true', 
                        help="data normalization store true" )
    parser.add_argument('--CHECK_POINT', type=str, default=None,
                        help="checkpoint path")
    parser.add_argument('--WEIGHT_DECAY', type=float,default= None,
                        help="The factor of L2 penalty on network weights")
    parser.add_argument('--BATCH_SIZE', type=int,default= None,
                        help="Batch size (optional, if absent will be set by the model")
    parser.add_argument('--TEST_BATCH', type=int,default= None,
                        help="Batch size of test (optional, if absent will be equal to train batch size")
    parser.add_argument('--LR', type=float, default= None,
                        help="Learning rate, set by the model if not specified.")
    parser.add_argument('--PATCH_SIZE', type=int, default= None,
                        help="Size of the spatial neighbourhood (optional, if "
                        "absent will be set by the model)")
    parser.add_argument('--ATT_MODE',  choices=['LCA','CBAM','CAM','SAM','Base'],default='LCA',help="select attention mechanism for SFCN (default LCA)")
    parser.add_argument('--MLWBDN-LEVEL', type=int,default=3,help="Level of scales (default: 1)" )
    parser.add_argument('--RUNS', type=int, default=1, help="Number of runs (default: 1)")
    parser.add_argument('--ENV_NAME', type=str, default='exp',
                        help="env name of visdom, for paremeter optimization vision")
    parser.add_argument('--AMP', action='store_true',
                        help="use automatic maxed precision to save GPU memory during training")
    parser.add_argument('--VISDOM',  type=int, default=1,
                        help="use visdom for training visual(default 1 for use visdom)")
    parser.add_argument('--OUT_MORE',  action= 'store_true',
                        help="Save middle outputs of SSRNet")
    
    
    args = vars(parser.parse_args())
    return args