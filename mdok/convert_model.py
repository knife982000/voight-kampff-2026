import safetensors.torch

tensors = safetensors.torch.load_file("mdok/classifier_model/adapter_model.safetensors")

out = {}
for key, value in tensors.items():
    if "word_embeddings" in key:
        continue
    new_key = key.replace("base_model.model.", "")
    if new_key.startswith("classifier"):
        #out[new_key] = value.clone()
        new_key = new_key.replace("classifier", "classifier.modules_to_save.default")
    out[new_key] = value

safetensors.torch.save_file(out, "mdok/classifier_model/adapter_model.safetensors")