# import torch
# from models.TinyFlowNet import TinyFlowNet
# import onnx
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# example_input = torch.randn(1, 6, 60, 60, device="cpu") # exmample for the forward pass input
# pytorch_model = TinyFlowNet()
# ONNX_PATH="tinyflownet.onnx"
#
# torch.onnx.export(
#     model=pytorch_model,
#     args=example_input,
#     f=ONNX_PATH, # where should it be saved
#     verbose=False,
#     export_params=True,
#     do_constant_folding=False,  # fold constant values for optimization
#     # do_constant_folding=True,   # fold constant values for optimization
#     input_names=['input'],
#     output_names=['output']
# )
# onnx_model = onnx.load(ONNX_PATH)
# onnx.checker.check_model(onnx_model)
#
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # dummy_input = torch.randn(1, 60, 60, 6, device=device)
# #
# # network_data = torch.load('model_best.pth.tar')
# # model = TinyFlowNet()
# #
# # model.load_state_dict(network_data['state_dict'])
# # model.eval()
# # torch.onnx.export(model.module, dummy_input, "tinyflownet.onnx", export_params = True)
# # torch.onnx.export(model, dummy_input, "tinyflownet.onnx", verbose=True, input_names=['input_1'], output_names=['output_1'])