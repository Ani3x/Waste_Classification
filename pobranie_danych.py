from roboflow import Roboflow

rf = Roboflow(api_key="")

project = rf.workspace("nazwa_workspaceu").project("nazwa_projektu")

dataset = project.version(1).download("folder")

print("Dataset został pobrany do folderu:", dataset.location)
