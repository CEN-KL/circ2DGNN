import subprocess
import logging
import datetime

log_path = './logs/' 
log_name = 'log_grid_search_' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.log'
logging.basicConfig(filename=log_path + log_name, level=logging.DEBUG)
logging.info('Script starts at : {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

k_neg_training = [1, 2, 3, 4, 5]
k_neg_testing = [1, 2, 3, 4, 5]

logging.info("learning rate")
for neg_test in k_neg_testing:
    cmd_for_test_neg_sample = f"python src/data_split.py --k_neg={neg_test}"
    logging.info(f"=========== 🍉 测试集负采样比例 = 1 : {neg_test} 🍉 ===========")
    logging.info(f"command: {cmd_for_test_neg_sample}")
    logging.info("runs at {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    try:
        res = subprocess.run(cmd_for_test_neg_sample, shell=True)
        if res.returncode == 0:
            logging.info("command: " + cmd_for_test_neg_sample)
            logging.info("Ends at {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        else: 
            logging.warning(f"Error !")
            continue
    except subprocess.CalledProcessError as e:
        logging.warning(f"Command failed with return code {e.returncode}, error message: {e.stderr}\n")
        continue  # 继续下一个参
    
    for neg_train in k_neg_training:
        command = f"python src/train.py --cuda=1 --k_neg={neg_train} --k_neg_test={neg_test}"
        logging.info(f"=========== 🍌 训练集负采样比例 = 1 : {neg_train} 🍌 ===========")
        logging.info(f"command: {command}")
        logging.info("Starts at {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        try:
            res = subprocess.run(command, shell=True)
            if res.returncode == 0:
                logging.info("Ends at {}\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            else: 
                logging.warning(f"Error !")
        except subprocess.CalledProcessError as e:
            logging.warning(f"Command failed with return code {e.returncode}, error message: {e.stderr}\n")
            continue  # 继续下一个参
        
    logging.info(f"=========== 🍉 测试集负采样比例 = 1 : {neg_test} 🍉 ===========\n")

    
logging.info('Script ends at : {}'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))