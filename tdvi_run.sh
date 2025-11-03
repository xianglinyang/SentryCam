#! /bin/bash


# ###############################################################
# content_path="/data2/xianglin/SentryCam/ResNet_CIFAR100_UNSTABLE"

# resumes=($(seq 0 1 39))
# iterations=($(seq 1 1 40))
# for i in {0..39}; 
# do
#     nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
# done

###############################################################
# content_path="/data2/xianglin/SentryCam/ResNet_CIFAR100"

# resumes=($(seq 0 1 39))
# iterations=($(seq 1 1 40))
# for i in {0..39}; 
# do
#     CUDA_VISIBLE_DEVICES=1 nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
# done

# ##############################################################
# content_path="/mnt/hdd1/ljiahao/xianglin/SentryCam/ResNet_CIFAR10_NONE_CF"

# resumes=($(seq 0 1 34))
# iterations=($(seq 1 1 35))
# for i in {0..34}; 
# do
#     nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 
# done

# ##############################################################
# content_path="/mnt/hdd1/ljiahao/xianglin/SentryCam/ResNet_CIFAR10_EWC_CF"

# resumes=($(seq 0 1 34))
# iterations=($(seq 1 1 35))
# for i in {0..34}; 
# do
#     nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 
# done

##############################################################
content_path="/mnt/hdd1/ljiahao/xianglin/SentryCam/ResNet_CIFAR10_ER_CF"

resumes=($(seq 0 1 34))
iterations=($(seq 1 1 35))
for i in {0..34}; 
do
    nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 
done
# ###############################################################
# content_path="/home/yangxl21/DVI_data/ResNet_CIFAR10"

# resumes=()
# # cold start
# resumes1=($(seq 0 1 9))
# # parallel run
# resumes2=(9)
# resumes3=($(seq 9 1 197))
# resumes+=( "${resumes1[@]}" "${resumes2[@]}" "${resumes3[@]}")
# iterations=($(seq 1 1 200))
# for i in {0..199}; 
# do
#     nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
# done
###############################################################
# content_path="/home/yangxl21/DVI_data/ResNet_FOOD101"

# resumes=($(seq 0 1 19))
# iterations=($(seq 1 1 20))
# for i in {0..19}; 
# do
#     nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
# done
###############################################################
# content_path="/home/yangxl21/DVI_data/ViT_CIFAR10"

# resumes=($(seq 0 1 199))
# iterations=($(seq 1 1 200))

# for i in {0..199}; 
# do
#     nohup nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
# done
###############################################################
# content_path="/data2/xianglin/SentryCam/ViT_CIFAR100"

# resumes=($(seq 0 1 199))
# iterations=($(seq 1 1 200))

# for i in {0..199}; 
# do
#     nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
# done
###############################################################
# content_path="/home/yangxl21/DVI_data/ViT_FOOD101"

# resumes=($(seq 0 1 19))
# iterations=($(seq 1 1 20))

# for i in {0..19}; 
# do
#     nohup python tdvi.py --content_path ${content_path} -i ${iterations[$i]} -r ${resumes[$i]} 1>> ${content_path}/tdvi_log 2>&1
# done
###############################################################