'''
    公司：芯源科技技术有限责任公司
创建日期：2022-02-18
  创建者：吕锋
    版本：2.0.0
  修订者：吕锋
修订日期：2022-04-18
    描述：本脚本用于将keras模型的.h5文件转换为tensorflow的.pb

'''
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

parser = argparse.ArgumentParser(description='convert the model file from format h5 to format pb.')
parser.add_argument('-h5_model',default="weight/model2.h5", help='Path to h5 model file.')
parser.add_argument('-pb_model',default="weight/model2.pb",help='Path to pb model file.')



def convert_h5_to_pb(h5_model_name,pb_model_name):
    model = tf.keras.models.load_model(h5_model_name,compile=False)  
    full_model = tf.function(lambda inputs: model(inputs))
    tensor_spec=[tf.TensorSpec(tensor.shape,tensor.dtype) for tensor in model.inputs]
<<<<<<< HEAD
    full_model = full_model.get_concrete_function(tensor_spec)
=======
    full_model = full_model.get_concrete_function(*tensor_spec)
>>>>>>> 1ef4b824c143e21a51fcb1147e98c2e186476d3c

    frozen_func = convert_variables_to_constants_v2(full_model)
    graph_def=frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
          print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    for node in graph_def.node:
      if node.op == 'DepthwiseConv2dNative':
        del node.attr['explicit_paddings']
      elif node.op=='FusedBatchNormV3':
        del node.attr['exponential_avg_factor']

    tf.io.write_graph(graph_or_graph_def=graph_def,
                      logdir=".",
                      name=pb_model_name,
                      as_text=False) 

if __name__ == '__main__':
    args=parser.parse_args()
    convert_h5_to_pb(args.h5_model,args.pb_model)
    print("done!")
