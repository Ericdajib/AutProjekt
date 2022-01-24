from threading import Thread

from kafka_test import KafkaConsumer
import os.path
import time
import pandas as pd
import sys
import _thread


class StopThreadException(Exception):
    def __init__(self, err='stop thread'):
        Exception.__init__(self, err)


class IatConsumer:
    def __init__(self, bs, topic):
        self.consumer = KafkaConsumer(topic, bootstrap_servers=bs, auto_offset_reset='smallest', consumer_timeout_ms=6000)
        self.output_path = './output'
        self.topic = topic
        self.record = ''
        self.flag = True

    def write_to_csv(self, num, send_time, value):
        dataframe = pd.DataFrame({'no.': num, 'send_time': send_time, 'value': value})
        zeit = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        output_file_name = str(self.topic) + '_' + zeit + '.csv'
        dataframe.to_csv(os.path.join(self.output_path, output_file_name), index=False, sep=',')
        print('\n[INFO] .csv file be created: ', output_file_name)

    def get_true_msg(self):
        for count, msg in enumerate(self.consumer):
            return count, msg.value_current.decode('utf8').split('&')

    def get_msg(self):
        print('[INFO] get_msg run')
        num, send_time, value = [], [], []
        try:
            while True:
                c, msg = self.get_true_msg()
                count = int(msg[0])
                self.record = str(count)
                num.append(msg[0])
                send_time.append(msg[1])
                value.append(msg[2])

                if count % 4000 == 0 and count != 0:
                    self.write_to_csv(num, send_time, value)
                    num, send_time, value = [], [], []
                # if count % 1000 == 0:
                #     print(msg)
        except TypeError:
            self.write_to_csv(num, send_time, value)
        print("[INFO] num:      ", len(num), sys.getsizeof(num))
        print("[INFO] send_time:", len(send_time), sys.getsizeof(send_time))
        print("[INFO] value:    ", len(value), sys.getsizeof(value))

    def set_flag(self):
        self.flag = False
        # raise StopThreadException()

    def record_data_point(self):
        print('[INFO] record_data_point run')
        while 1:
            file = open('./output/record.txt', 'w')
            input_str = input("Enter your input: ")
            print('[INFO] Received input is: ', input_str)
            if input_str == 'exit' or input_str == 'exit()':
                file.close()
                self.flag = False
                return
            elif input_str == 's' or input_str == 'save':
                file.write(str(self.record) + '\n')
                print('count: ', self.record)

    def run(self):
        try:
            _thread.start_new_thread(self.get_msg, ())
            _thread.start_new_thread(self.record_data_point, ())
        except Exception as e:
            print(e)

        while self.flag:
            pass


if __name__ == '__main__':
    print('[INFO] -------------------- Consumer run! --------------------')
    IatConsumer('localhost:9092', 'Current').run()
    print('[INFO] -------------------- Consumer done! --------------------')


