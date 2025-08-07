from src.utils.specs import Retrieve

device = Retrieve.get_device()
model = Retrieve.get_model(device)
model.eval()