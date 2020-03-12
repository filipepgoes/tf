import tensorflow as tf
print(tf.version)

string=tf.Variable("this is a string", tf.string)
number=tf.Variable(324, tf.int16)
floating=tf.Variable(3.567, tf.float64)

print(string)
print(number)
print(floating)

rank1_tensor=tf.Variable(["Test", "Ok", "Tim"], tf.string)
rank2_tensor=tf.Variable([["test","ok"],["test","yes"]], tf.string)

print(rank1_tensor)
print(rank2_tensor)
print(tf.rank(rank2_tensor))
print(rank2_tensor.shape)

tensor1=tf.ones([1,2,3])
tensor2=tf.reshape(tensor1, [2,3,1])
tensor3=tf.reshape(tensor2, [3,-1]) # -1 tells the tensor to calculate the dimension of the tensor in that place

print(tensor1)
print(tensor2)
print(tensor3)

t=tf.zeros([5,5,5,5])
print(t)
t=tf.reshape(t, [625])
print(t)
t=tf.reshape(t, [125,-1])
print(t)

