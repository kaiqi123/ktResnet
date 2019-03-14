self.is_training = tf.placeholder(tf.bool, name='MODE')
#first use:
self.img_bn1 = tf.cond(self.is_training,
lambda: batch_norm(self.img_fc1, is_training=self.is_training, center=True, scale=True, activation_fn=None, decay=0.9, scope='discriminator/img_bn1', reuse = False),
lambda: batch_norm(self.img_fc1, is_training=self.is_training, center=True, scale=True, activation_fn=None, decay=0.9, scope='discriminator/img_bn1', reuse = True))

# add update_ops before second ruse, and filter out unrelated update_ops(unrelated moving mean and variance)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
print('update_ops')
for key in update_ops:
print(key)
i2t_update_extra_ops = [elem for elem in update_ops if 'text_feature/attention' not in elem.name]

# second use:
self.img_neg_bn1 = batch_norm(self.img_neg_fc1, is_training=self.is_training, center=True, scale=True, activation_fn=None, decay=0.9, scope='discriminator/img_bn1', reuse = True)

# weight update and dependent extra_ops(moving mean and variance)
self.i2t_optimizer = tf.train.GradientDescentOptimizer(learning_rate )
i2t_update_grads = self.i2t_optimizer.minimize(self.i2t_loss)

i2t_train_ops = [i2t_update_grads] + i2t_update_extra_ops
self.i2t_updates = tf.group(*i2t_train_ops)