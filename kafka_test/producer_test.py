import time
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=["localhost:9092"])
time.sleep(3)

Q = [1, 2, 3, 2, 1]
S = [1, 2, 1, 2, 3, 2, 1, 3, 2, 1, 2, 1, 2, 3, 2, 1, 3, 2, 1, 1, 1, 2, 3, 2, 1]
print('producer start!')
for idx, elm in enumerate(S):
    time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    msg = str(idx) + '&' + str(time_now) + '&' + str(elm)
    producer.send('kafkatest', value=msg.encode('utf-8'))
    time.sleep(1)

producer.close()
