def choose_dataset(dataset_name):
    Dataset = None
    if dataset_name == "cityscapes_GTA":
        from .datasets_city import Dataset
    if dataset_name == "shoes":
        from .datasets_shoes import Dataset

    assert Dataset is not None

    return Dataset
