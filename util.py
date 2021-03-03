import os
import numpy as np
import shutil
import torch
import cv2



def save_checkpoint(state, is_best, save_path, model, dummy_input, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))
        # model_quantized = torch.quantization.convert(model.cpu())
        # model.cuda()
        input_names = ['input_0']
        output_names = ['output_0']
        torch.onnx.export(model.module, dummy_input, os.path.join(save_path,"tinyflownet.onnx"), verbose=False,
                          input_names=input_names, output_names=output_names, export_params=True, keep_initializers_as_inputs=True, opset_version=11)#, export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
        # torch.onnx.export(model_quantized.module, dummy_input, os.path.join(save_path, "tinyflownet_q.onnx"), verbose=False,
        #                   input_names=input_names, output_names=output_names)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map.detach().cpu().numpy()
    _, height, width = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    mag = np.zeros((height, width))
    ang = np.zeros((height, width))
    for i in range(0, height):
        for j in range(0, width):
            magnitude, angle = cart2pol(flow_map_np[0][i][j], flow_map_np[1][i][j])
            # print(flow[0][i][j], flow[1][i][j], magnitude, angle)
            mag[i][j] = magnitude
            ang[i][j] = angle

    mag = mag.astype(np.float32)
    ang = ang.astype(np.float32)
    # mag, ang = cv2.cartToPolar(flow_uv[0], flow_uv[1])

    hsv = np.zeros((height, width, 3)).astype(np.float32)
    hsv[..., 1] = 1
    hsv[..., 0] = (ang * 90 / (np.pi) + 90) * 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    # bgr = cv2.flip(bgr, 0)
    # bgr = cv2.resize(bgr, (width * 20, height * 20), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

    # rgb_map = np.ones((3,h,w)).astype(np.float32)
    # if max_value is not None:
    #     normalized_flow_map = flow_map_np / max_value
    # else:
    #     normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    # rgb_map[0] += normalized_flow_map[0]
    # rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    # rgb_map[2] += normalized_flow_map[1]
    return bgr.clip(0,1)
