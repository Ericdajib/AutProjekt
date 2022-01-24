from kafka import KafkaConsumer


def consumer_demo():
    consumer = KafkaConsumer("kafkatest", bootstrap_servers=["localhost:9092"], auto_offset_reset='latest', consumer_timeout_ms=6000)
    S = []
    for msg in consumer:
        msg = msg.value.decode(encoding="utf-8")
        msg = msg.split("&")  # list [1, 2021-1-10-20:00:22, 5] N, time, value

        if len(msg) == 3:
            N = msg[0]
            t = msg[1]
            st = int(msg[2])
            print('N, t, st', N, t, st)

            if len(S) <= 5:
                S.append(st)
            else:
                S[:-1] = S[1:]  # 数据整体右移
                S[-1] = st

            print('S', S)


