
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'fcgan':
        assert (opt.dataset_mode == 'single')
        from .fcgan_model import FCGANModel
        model = FCGANModel()
    elif opt.model == 'cgan':
        from .cgan_model import CGANModel
        model = CGANModel()
    elif opt.model == 'cgan2':
        from .cgan2_model import CGANModel
        model = CGANModel()
    elif opt.model == 'cgan_cycle':
        from .cgan_cycle_model import CGANCycleModel
        model = CGANCycleModel()
    elif opt.model == 'cgan2_cycle':
        from .cgan2_cycle_model import CGANCycleModel
        model = CGANCycleModel()
    elif opt.model == 'twostage':
        from .twostage_model import TwoStageModel
        model = TwoStageModel()
    elif opt.model == 'twostage_cycle':
        from .twostage_cycle_model import TwoStageCycleModel
        model = TwoStageCycleModel()
    elif opt.model == 'twostage_factd':
        from .twostage_factD_model import TwoStageModel
        model = TwoStageModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'segmentation':
        from .segm_model import SegmentationModel
        model = SegmentationModel()
    elif opt.model == 'segmentation_cycle':
        from .segm_cycle_model import SegmentationCycleModel
        model = SegmentationCycleModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
