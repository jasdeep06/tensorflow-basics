import tensorflow as tf

filename_queue=tf.train.string_input_producer(["/Users/jasdeepsinghchhabra/Desktop/csvread.csv"])

reader=tf.TextLineReader()
key,value=reader.read(filename_queue)

record_defaults=[[1.],[1.],[1.],[1.]]
col1,col2,col3,col4=tf.decode_csv(value,record_defaults=record_defaults)
features=tf.pack([col1,col2,col3])

with tf.Session() as sess:
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(coord=coord)
    for i in range(3):
        example,label=sess.run([features,col4])
        print(example)
        print(label)
    coord.request_stop()
    coord.join(threads)