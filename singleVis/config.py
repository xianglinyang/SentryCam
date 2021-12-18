dataset_config = {
    "cifar10": {
        "TRAINING_LEN": 50000,
        "TESTING_LEN": 10000,
        "LAMBDA":10.,
        "DOWNSAMPLING_RATE": .1,
        "L_BOUND":0.6,
        "training_config":{
            "EPOCH_NUM": 6,
            "TIME_STEPS": 11,
            "TEMPORAL_PERSISTENT": 1,
            "NUMS": 5,   # the number of epochs to go through in one go
            "PATIENT": 3,    # early stopping patient
        }
        
    },
    "mnist": {
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":1.,
        "DOWNSAMPLING_RATE": .2,
        "L_BOUND":0.4,
        "training_config":{
            "EPOCH_NUM": 6,
            "TIME_STEPS": 10,
            "TEMPORAL_PERSISTENT": 1,
            "NUMS": 5,   # the number of epochs to go through in one go
            "PATIENT": 3,    # early stopping patient
        }
    },
    "fmnist": {
        "TRAINING_LEN": 60000,
        "TESTING_LEN": 10000,
        "LAMBDA":20.,
        "DOWNSAMPLING_RATE": .1,
        "L_BOUND":0.5,
        "training_config":{
            "EPOCH_NUM": 6,
            "TIME_STEPS": 11,
            "TEMPORAL_PERSISTENT": 1,
            "NUMS": 5,   # the number of epochs to go through in one go
            "PATIENT": 3,    # early stopping patient
        }
    }
}