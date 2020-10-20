from django.http import HttpResponse
from django.shortcuts import render
import os,sys
from django.http import JsonResponse
from tensorflow_model.testAPI import testEnterFunction
destination_path = '/home/liu/chenhaoran/manipulation_detection_web/upload_dir/input'
detection_save_path = '/home/liu/chenhaoran/manipulation_detection_web/upload_dir/output'
from get_score import GetScore
"""
global variable
"""
# model_path = '/home/liu/libiao/Simple_net_output/checkpoints/8.20_new_out/checkpoint.61-0.0452-0.9976-0.9341-0.8720-0.9004.hdf5'
model_path = 'tensorflow_model/checkpoint/checkpoint.61-0.0452-0.9976-0.9341-0.8720-0.9004.hdf5'

def upload_file(request):
    print('start to run the function upload_file')
    if request.method == "POST":    # 请求方法为POST时，进行处理
        myFile =request.FILES.get("myfile", None)    # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return HttpResponse("no files for upload!")
        destination = open(os.path.join(destination_path,myFile.name),'wb+')    # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():      # 分块写入文件
            destination.write(chunk)
        destination.close()

        input_img_path = os.path.join(destination_path, myFile.name)
        input_img_path = os.path.join('..',input_img_path)
        output_dir = os.path.join('..',detection_save_path)
        output_dir = os.path.join(output_dir,myFile.name)
        detection_flag = detection(input_img_path,output_dir)
        # using output_dir to generate prediction score


        print('the output:', output_dir)
        if detection_flag:
            score = GetScore(output_dir).basic_description()
            score = str(round(score,4))
            res = {'score', score, 'input_img_name', myFile.name,'pred_img_name',myFile.name.replace('.jpg', '.png') }
            return JsonResponse(res,json_dumps_params={'ensure_ascii': False})
        else:
            return HttpResponse("error")

def upload_page(request):
    print('start to run uplaod_page function')
    return render(request,'uploadPage.html')


def detection(input_dir,output_dir):
    print('start to run press_detection_button function')
    # running forward process
    print('开始进入testEnterFunction')
    testEnterFunction(input_dir=input_dir, output_dir=output_dir)
    print('find loc')
    print(input_dir)
    print(output_dir)
    if os.path.exists(output_dir) or True:
        print('预测文件存在，推理阶段成功')
        # 开始返回到前端页面
        return True
    else:
        print('推理阶段出错，请重新测试')
        return False