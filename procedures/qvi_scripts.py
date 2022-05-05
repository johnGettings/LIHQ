import configparser

def qvi_config(frame_int):
    config = configparser.ConfigParser()
    config.testset_root = './output/GPEN/Round2'
    config.test_size = (1024, 1024)
    config.test_crop_size = (1024, 1024)

    config.mean = [0.429, 0.431, 0.397]
    config.std  = [1, 1, 1]

    config.inter_frames = frame_int


    config.model = 'QVI'
    config.pwc_path = './QVI/utils/pwc-checkpoint.pt'


    config.store_path = './output/QVI/'
    config.checkpoint = './QVI/qvi_release/model.pt'
    
    return(config)