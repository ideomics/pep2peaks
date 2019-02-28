import numpy as np
import tensorflow as tf
aa=np.array([
        [[ 0,  0,  2,  3],
       [ 1,  0,  6,  7],
       [ 0,  1, 10, 11]],

 [[ 0,  1,  2,  3],
       [ 0,  0,  6,  7],
       [ 1,  0, 10, 11]]
        ])
print(aa.shape)
aa1=aa[:,:,:2]
aa2=aa[:,:,2:]
print(aa1)
print(aa1.shape)
print(aa2)
print(aa2.shape)

bb1=tf.constant(aa1)
bb2=tf.constant(aa2)
print(bb1.get_shape())
print(bb2.get_shape())

cc=tf.concat([bb1,bb2],axis=2)
with tf.Session() as sess:
    print(sess.run(bb1))
    print(sess.run(bb2))
    print(sess.run(cc))