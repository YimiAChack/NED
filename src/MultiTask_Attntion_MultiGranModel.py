# encoding=utf-8

import numpy as np
import tensorflow as tf


class MultiTask_MultiGranModel(object):
    def _conv(self, name, in_, ksize, reuse=False):
        num_filters = ksize[3]

        with tf.variable_scope(name, reuse=reuse) as scope:
            # different CNN for different views
            W = tf.Variable(tf.truncated_normal(ksize, stddev=0.1), name="W")
            biases = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

            # same CNN for different views
            # W = tf.get_variable("weights", ksize, initializer=tf.contrib.layers.xavier_initializer())
            # W = tf.get_variable("weights", ksize, initializer=tf.truncated_normal_initializer(stddev=0.1))
            # biases = tf.get_variable("biases", [num_filters], initializer=tf.constant_initializer(0.1))

            conv = tf.nn.conv2d(in_, W, strides=[1, 1, 1, 1], padding="VALID")
            h = tf.nn.relu(tf.nn.bias_add(conv, biases), name=scope.name)
        return h

    def _maxpool(self, name, in_, ksize, strides):
        pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides, padding='VALID', name=name)
        #print name, pool.get_shape().as_list()
        return pool

    def __init__(self, max_len1, max_len2, filter_sizes, pool_sizes, filter_sizes2, pool_sizes2, num_filters,disease_num,operation_num,
                 l2_reg_lambda=0.0, constraint_lambda=0.0, alpha=0.5, type_CNN=2, view_num=0, view_nums=[],batch_size=128):
        channel_num = 4
        self.sec_disease_num = disease_num -1
        self.sec_operation_num = operation_num -1

		#-------------------------------------------
		#Placeholders for input, output and dropout
        self.input_tensor_d = tf.placeholder(tf.float32, [None, max_len1, max_len1, channel_num], name="input_tensor_main_description") #主诊断
        self.input_tensor_o = tf.placeholder(tf.float32, [None, max_len2, max_len2, channel_num], name="input_tensor_main_operation") #主手术
        input_tensor_sec_tmp = tf.placeholder(tf.float32, [None, max_len1, max_len1, channel_num])
        self.input_tensor_sec_d = tf.stack([input_tensor_sec_tmp]*self.sec_disease_num,name="input_tensor_descriptions")
        self.input_tensor_sec_o = tf.stack([input_tensor_sec_tmp]*self.sec_operation_num,name="input_tensor_operations")

        #empty mask
        mask_sec_tmp = tf.placeholder(tf.float32, [None,2])
        self.mask_sec_d = tf.stack([mask_sec_tmp]*self.sec_disease_num, name="mask_sec_d")
        self.mask_sec_o = tf.stack([mask_sec_tmp]*self.sec_operation_num, name="mask_sec_o")
        # self.mask_sec_d = tf.placeholder(tf.float32,[None, self.sec_disease_num], name="mask_sec_d")
        # self.mask_sec_o = tf.placeholder(tf.float32,[None, self.sec_operation_num], name="mask_sec_o")

        self.input_y_d = tf.placeholder(tf.float32, [None,2], name="input_y_main_description")
        self.input_y_o = tf.placeholder(tf.float32, [None,2], name="input_y_main_operation")
        input_sec_tmp = tf.placeholder(tf.float32, [None,2]) 
        self.input_y_sec_d = tf.stack([input_sec_tmp]*self.sec_disease_num,name = "input_y_descriptions")
        self.input_y_sec_o = tf.stack([input_sec_tmp]*self.sec_operation_num,name = "input_y_operations")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        tmp_cor = tf.placeholder(tf.float32, [None,1])
        self.matrix = tf.stack([tmp_cor] *(disease_num - 1 + operation_num), name="cooccurence")
        self.constraint_lambda = constraint_lambda


        # Keeping track of l2 regularization loss (optional)
        l2_loss_d = tf.constant(0.0)
        l2_loss_sec_d = tf.constant(0.0)
        l2_loss_o = tf.constant(0.0)
        l2_loss_sec_o = tf.constant(0.0)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_d = list()
        pooled_outputs_o = list()
        pooled_outputs_sec_d = list()
        pooled_outputs_sec_o = list()


        input_tensor_d = tf.expand_dims(self.input_tensor_d, 4)  # N x W x H x V  => N x W x H x V x C
        input_tensor_d = tf.transpose(input_tensor_d,
                                    perm=[3, 0, 1, 2, 4])  # N x W x H x V x C =>  V x N x W x H x C
        input_tensor_sec_d = tf.stack([input_tensor_d]*self.sec_disease_num)

        input_tensor_o = tf.expand_dims(self.input_tensor_o, 4)  # N x W x H x V  => N x W x H x V x C
        input_tensor_o = tf.transpose(input_tensor_o,
                                              perm=[3, 0, 1, 2, 4])  # N x W x H x V x C =>  V x N x W x H x C
        input_tensor_sec_o = tf.stack([input_tensor_o]*self.sec_operation_num)


        if type_CNN == 2: # multi-view
            with tf.name_scope("CNN_Description"):
                view_c_num = 0
                for i in range(channel_num):
                    # set reuse True for i > 0, for weight-sharing
                    reuse_f = (i != 0)
                    with tf.variable_scope("CNN_Description", reuse=reuse_f):
                        if len(view_nums) != 0:
                            if len(view_nums) <= view_c_num or view_nums[view_c_num] != i:
                                continue
                            else:
                                view_c_num += 1
                        #print("AHAA" + str(i) + "\n")
                        view = tf.gather(input_tensor_d, i)  # N x W x H x C

                        filter_shape1 = [filter_sizes[0], filter_sizes[0], 1, num_filters]
                        filter_shape2 = [filter_sizes[1], filter_sizes[1], num_filters, num_filters * 2]
                        p_size1 = [1, pool_sizes[0], pool_sizes[0], 1]
                        p_size2 = [1, pool_sizes[1], pool_sizes[1], 1]

                        conv1 = self._conv('conv1', view, filter_shape1, reuse=reuse_f)
                        pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])

                        conv2 = self._conv('conv2', pool1, filter_shape2, reuse=reuse_f)
                        pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

                        dim1 = np.prod(pool2.get_shape().as_list()[1:])
                        reshape = tf.reshape(pool2, [-1, dim1])

                        pooled_outputs_d.append(reshape)

            with tf.name_scope("CNN_sec_Descriptions"): #副诊断的
                for k in range(self.sec_disease_num):
                    # if self.mask_sec_d[k]:#???????????need test
                    #     pooled_outputs_sec_d.append([])
                    #     continue
                	pooled_outputs_sec_d_tmp = list()
                	view_c_num = 0
	                for i in range(channel_num):
	                    # set reuse True for i > 0, for weight-sharing
	                    reuse_f = (i != 0)
	                    with tf.variable_scope("CNN_Description", reuse=reuse_f):
	                        if len(view_nums) != 0:
	                            if len(view_nums) <= view_c_num or view_nums[view_c_num] != i:
	                                continue
	                            else:
	                                view_c_num += 1
	                        #print("AHAA" + str(i) + "\n")
	                        view = tf.gather(input_tensor_sec_d[k], i)  # N x W x H x C

	                        filter_shape1 = [filter_sizes[0], filter_sizes[0], 1, num_filters]
	                        filter_shape2 = [filter_sizes[1], filter_sizes[1], num_filters, num_filters * 2]
	                        p_size1 = [1, pool_sizes[0], pool_sizes[0], 1]
	                        p_size2 = [1, pool_sizes[1], pool_sizes[1], 1]

	                        conv1 = self._conv('conv1', view, filter_shape1, reuse=reuse_f)
	                        pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])

	                        conv2 = self._conv('conv2', pool1, filter_shape2, reuse=reuse_f)
	                        pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

	                        dim1_sec = np.prod(pool2.get_shape().as_list()[1:])
	                        reshape = tf.reshape(pool2, [-1, dim1_sec])

	                        pooled_outputs_sec_d_tmp.append(reshape)
	            	pooled_outputs_sec_d.append(pooled_outputs_sec_d_tmp)


            with tf.name_scope("CNN_Operation"):
                view_c_num = 0
                for i in range(channel_num):
                    # set reuse True for i > 0, for weight-sharing
                    reuse_f = (i != 0)

                    with tf.variable_scope("CNN_Operation", reuse=reuse_f):
                        if len(view_nums) != 0:
                            if len(view_nums) <= view_c_num or view_nums[view_c_num] != i:
                                continue
                            else:
                                view_c_num += 1
                        #print("AHAA" + str(i) + "\n")
                        view = tf.gather(input_tensor_o, i)  # N x W x H x C

                        filter_shape1 = [filter_sizes2[0], filter_sizes2[0], 1, num_filters / 2]
                        filter_shape2 = [filter_sizes2[1], filter_sizes2[1], num_filters / 2, num_filters]
                        p_size1 = [1, pool_sizes2[0], pool_sizes2[0], 1]
                        p_size2 = [1, pool_sizes2[1], pool_sizes2[1], 1]

                        conv1 = self._conv('conv1', view, filter_shape1, reuse=reuse_f)
                        pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])

                        conv2 = self._conv('conv2', pool1, filter_shape2, reuse=reuse_f)
                        pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

                        dim2 = np.prod(pool2.get_shape().as_list()[1:])
                        reshape = tf.reshape(pool2, [-1, dim2])

                        pooled_outputs_o.append(reshape)


            with tf.name_scope("CNN_sec_Operations"):
            	for k in range(self.sec_operation_num):
                    # if self.mask_sec_o[k]:#???????????need test
                    #     pooled_outputs_sec_d.append([])
                    #     continue
            		pooled_outputs_sec_o_tmp = list()
	                view_c_num = 0
	                for i in range(channel_num):
	                    # set reuse True for i > 0, for weight-sharing
	                    reuse_f = (i != 0)

	                    with tf.variable_scope("CNN_Operation", reuse=reuse_f):
	                        if len(view_nums) != 0:
	                            if len(view_nums) <= view_c_num or view_nums[view_c_num] != i:
	                                continue
	                            else:
	                                view_c_num += 1
	                        #print("AHAA" + str(i) + "\n")
	                        view = tf.gather(input_tensor_sec_o[k], i)  # N x W x H x C

	                        filter_shape1 = [filter_sizes2[0], filter_sizes2[0], 1, num_filters / 2]
	                        filter_shape2 = [filter_sizes2[1], filter_sizes2[1], num_filters / 2, num_filters]
	                        p_size1 = [1, pool_sizes2[0], pool_sizes2[0], 1]
	                        p_size2 = [1, pool_sizes2[1], pool_sizes2[1], 1]

	                        conv1 = self._conv('conv1', view, filter_shape1, reuse=reuse_f)
	                        pool1 = self._maxpool('pool1', conv1, ksize=p_size1, strides=[1, 1, 1, 1])

	                        conv2 = self._conv('conv2', pool1, filter_shape2, reuse=reuse_f)
	                        pool2 = self._maxpool('pool2', conv2, ksize=p_size2, strides=[1, 1, 1, 1])

	                        dim2_sec = np.prod(pool2.get_shape().as_list()[1:])
	                        reshape = tf.reshape(pool2, [-1, dim2_sec])

	                        pooled_outputs_sec_o_tmp.append(reshape)

	                pooled_outputs_sec_o.append(pooled_outputs_sec_o_tmp)



            view_num_len = len(pooled_outputs_d)
            # print("LEN:" + str(view_num_len))

            with tf.name_scope("Description_view_pooling"):
                x = tf.stack(pooled_outputs_d)  # 4 * N * 7744
                x = tf.transpose(x, perm=[1, 2, 0])  # N * 7744 * 4

                reshape = tf.reshape(x, [-1, view_num_len])
                #print reshape.get_shape().as_list()

                Weights = tf.Variable(tf.random_uniform([view_num_len, 1], 0.0, 1.0), name="W")

                y_d = tf.matmul(reshape, Weights, name="view_pooling")
                y_d = tf.reshape(y_d, [-1, dim1])
                #print y_d.get_shape().as_list()

            with tf.name_scope("sec_Descriptions_view_pooling"):
            	y_sec_d = list()
            	for i in range(self.sec_disease_num):
                    # if self.mask_sec_d[k]:#???????????need test
                    #     pooled_outputs_sec_d.append([])
                    #     continue
	                x = tf.stack(pooled_outputs_sec_d[i])  # 4 * N * 7744
	                x = tf.transpose(x, perm=[1, 2, 0])  # N * 7744 * 4

	                reshape = tf.reshape(x, [-1, view_num_len])
	                #print reshape.get_shape().as_list()

	                Weights = tf.Variable(tf.random_uniform([view_num_len, 1], 0.0, 1.0), name="W")

	                y_sec_d_tmp = tf.matmul(reshape, Weights, name="view_pooling")
	                y_sec_d_tmp = tf.reshape(y_sec_d_tmp, [-1, dim1])
	                y_sec_d.append(y_sec_d_tmp) #??append?
	                #print y_d.get_shape().as_list()

            with tf.name_scope("Operation_view_pooling"):
                x = tf.stack(pooled_outputs_o)  # 4 * N * 7744
                x = tf.transpose(x, perm=[1, 2, 0])  # N * 7744 * 4
                reshape = tf.reshape(x, [-1, view_num_len])
                # print reshape.get_shape().as_list()

                Weights = tf.Variable(tf.random_uniform([view_num_len, 1], 0.0, 1.0), name="W")

                y_o = tf.matmul(reshape, Weights, name="view_pooling")
                y_o = tf.reshape(y_o, [-1, dim2])
                #print y_o.get_shape().as_list()

            with tf.name_scope("sec_Operations_view_pooling"):
            	y_sec_o = list()
            	for i in range(self.sec_operation_num):
                    # if self.mask_sec_d[k]:#???????????need test
                    #     pooled_outputs_sec_d.append([])
                    #     continue
	                x = tf.stack(pooled_outputs_sec_o[i])  # 4 * N * 7744
	                x = tf.transpose(x, perm=[1, 2, 0])  # N * 7744 * 4
	                reshape = tf.reshape(x, [-1, view_num_len])
	                #print reshape.get_shape().as_list()

	                Weights = tf.Variable(tf.random_uniform([view_num_len, 1], 0.0, 1.0), name="W")

	                y_sec_o_tmp = tf.matmul(reshape, Weights, name="view_pooling")
	                y_sec_o_tmp = tf.reshape(y_sec_o_tmp, [-1, dim2])
	                y_sec_o.append(y_sec_o_tmp) #??append?
	                #print y_o.get_shape().as_list()

        else : 
           print '..'


        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop_d = tf.nn.dropout(y_d, self.dropout_keep_prob, name="hidden_output_description_drop")
            self.h_drop_o = tf.nn.dropout(y_o, self.dropout_keep_prob, name="hidden_output_operation_drop")
            self.h_drop_sec_d = list()
            self.h_drop_sec_o = list()
            for i in range(self.sec_disease_num):
                # if self.mask_sec_d[k]:#???????????need test
                #     self.h_drop_sec_d.append.append([])
                #     continue
            	self.h_drop_sec_d.append(tf.nn.dropout(y_sec_d[i], self.dropout_keep_prob, name="hidden_output_description_drop"))
            for i in range(self.sec_operation_num):
                # if self.mask_sec_d[k]:#???????????need test
                #     self.h_drop_sec_d.append.append([])
                #     continue
            	self.h_drop_sec_o.append(tf.nn.dropout(y_sec_o[i], self.dropout_keep_prob, name="hidden_output_operation_drop"))
            #print self.h_drop_d.get_shape().as_list()
            #print self.h_drop_o.get_shape().as_list()


        # Share Layer Construction
        with tf.name_scope("FC1"):
            dim = min(int(dim1 / 2), int(dim2 / 2)) #?????dim1_sec?dim2_sec
            # print ("dim1: " + str(dim1))
            # print ("dim2: " + str(dim2))
            # print ("FC DIM:" + str(dim) + "\n")

            W1 = tf.Variable(name="W1", initial_value=tf.truncated_normal(shape=[dim1, dim], stddev=0.1))
            b1 = tf.Variable(tf.constant(0.1, shape=[dim]), name="b1")

            self.fc_d = tf.nn.relu(tf.matmul(self.h_drop_d, W1) + b1) 
            self.fc_drop_d = tf.nn.dropout(self.fc_d, self.dropout_keep_prob)#疾病的FC(feature)

            self.fc_drop_sec_d = list()
            for i in range(self.sec_disease_num):
                # if self.mask_sec_d[k]:
                #     self.h_drop_sec_d.append.append([])
                #     continue
            	fc_d_tmp = tf.nn.relu(tf.matmul(self.h_drop_sec_d[i], W1) + b1) 
            	fc_drop_sec_d_tmp = tf.nn.dropout(fc_d_tmp, self.dropout_keep_prob)
            	self.fc_drop_sec_d.append(fc_drop_sec_d_tmp)

            W2 = tf.Variable(name="W2", initial_value=tf.truncated_normal(shape=[dim2, dim], stddev=0.1))
            b2 = tf.Variable(tf.constant(0.1, shape=[dim]), name="b2")

            self.fc_o = tf.nn.relu(tf.matmul(self.h_drop_o, W2) + b2) 
            self.fc_drop_o = tf.nn.dropout(self.fc_o, self.dropout_keep_prob)#手术的FC(feature)

            self.fc_drop_sec_o = list()
            for i in range(self.sec_operation_num):
            	fc_o_tmp = tf.nn.relu(tf.matmul(self.h_drop_sec_o[i], W2) + b2) 
            	fc_drop_sec_o_tmp = tf.nn.dropout(fc_o_tmp, self.dropout_keep_prob)
            	self.fc_drop_sec_o.append(fc_drop_sec_o_tmp)


        #Attention Layer Construction
        with tf.name_scope("Operation_Attention"):
            self.fc_drop_sec_o_attention = [0.0] * dim #这里取手术dim2、诊断dim1,最短的dim，进行attention和共享
            
            b = tf.Variable(tf.constant(0.1, shape=[5,dim]), name="b")
            for i in range(n):
                #attention公式待修改，尤其是考虑一下矩阵维度乘法的问题
                W = tf.Variable(name="W", initial_value=tf.truncated_normal(shape=[dim, dim], stddev=0.1))
                u = tf.nn.tanh(tf.matmul(tf.concat(self.fc_drop_sec_o[i], self.fc_drop_d), W) + b[i]) 
                a = tf.nn.softmax(u)
                # a = tf.constant(0.1)
                self.fc_drop_sec_o_attention += tf.multiply(self.fc_drop_sec_o[i],a)


        # Share Layer Construction
        #主诊断和所有attention后的手术共享权值
        with tf.name_scope("Multitask_main_o_d"): 
            #Shared_layer
            self.shared_layer = tf.add(alpha * self.fc_drop_d, (1 - alpha) * self.fc_drop_sec_o_attention, name="Shared_layer")
            # self.shared_layer = tf.div(tf.add(self.h_drop_d, self.h_drop_o), 2, name="Shared_layer")
            #print self.shared_layer.get_shape().as_list()

            W1 = tf.get_variable(name="tt1_W", shape=[dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            W2 = tf.get_variable(name="st1_W", shape=[dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            W3 = tf.get_variable(name="st2_W", shape=[dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            W4 = tf.get_variable(name="tt2_W", shape=[dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))

            #各自的FC和share_layer进行加权
            self.task1_r = tf.add(tf.multiply(self.shared_layer, W2), tf.multiply(self.fc_drop_d, W1),
                                  name="main_description_r")

            self.task2_r = tf.add(tf.multiply(self.shared_layer, W3), tf.multiply(self.fc_drop_o, W4),
                                  name="operation_r")
            self.task_sec_o = list()
            for i in range(self.sec_operation_num):
             	task_sec_o_tmp = tf.add(tf.multiply(self.shared_layer, W3), tf.multiply(self.fc_drop_sec_o[i], W4))
            	self.task_sec_o.append(task_sec_o_tmp)
            #print self.task1_r.get_shape().as_list()


        #主诊断和副诊断共享权值
        with tf.name_scope("Share_main_o_sec_d"): 
            W = tf.get_variable(name="tt2_W_sec_d", shape=[dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
            self.task_sec_d = list()
            for i in range(self.sec_disease_num):
                shared_layer_tmp  = tf.add(alpha * self.fc_drop_d, (1 - alpha) * self.fc_drop_sec_d[i])
                self.task_sec_d.append(tf.add(tf.multiply(shared_layer_tmp, W), tf.multiply(self.fc_drop_sec_d[i], W))) #W? 
           

        with tf.name_scope("FC2"):
            W1 = tf.Variable(name="W1", initial_value=tf.truncated_normal(shape=[dim, dim / 2], stddev=0.1))
            b1 = tf.Variable(tf.constant(0.1, shape=[dim / 2]), name="b1")

            self.task1_representation = tf.nn.relu(tf.matmul(self.task1_r, W1) + b1)
            self.task1_representtion = tf.nn.dropout(self.task1_representation, self.dropout_keep_prob)

            self.task_sec_d_representation = list()
            for i in range(self.sec_disease_num):
            	task_sec_d_representation_tmp = tf.nn.relu(tf.matmul(self.task_sec_d[i], W1) + b1)
            	self.task_sec_d_representation.append(tf.nn.dropout(task_sec_d_representation_tmp, self.dropout_keep_prob))


            W2 = tf.Variable(name="W2", initial_value=tf.truncated_normal(shape=[dim, dim / 2], stddev=0.1))
            b2 = tf.Variable(tf.constant(0.1, shape=[dim / 2]), name="b2")

            self.task2_representation = tf.nn.relu(tf.matmul(self.task2_r, W2) + b2)
            self.task2_representation = tf.nn.dropout(self.task2_representation, self.dropout_keep_prob)

            self.task_sec_o_representation = list()
            for i in range(self.sec_operation_num):
            	task_sec_o_representation_tmp = tf.nn.relu(tf.matmul(self.task_sec_o[i], W2) + b2)
            	self.task_sec_o_representation.append(tf.nn.dropout(task_sec_o_representation_tmp, self.dropout_keep_prob))



        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W_d = tf.get_variable(name="W_d", shape=[dim / 2, 2],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_d = tf.Variable(tf.constant(0.1, shape=[2]), name="b_d")

            l2_loss_d += tf.nn.l2_loss(W_d) #统计全链接层中的参数个数
            l2_loss_d += tf.nn.l2_loss(b_d)

            W_o = tf.get_variable(name="W_o", shape=[dim / 2, 2],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_o = tf.Variable(tf.constant(0.1, shape=[2]), name="b_o")

            l2_loss_o += tf.nn.l2_loss(W_o) #统计全链接层中的参数个数
            l2_loss_o += tf.nn.l2_loss(b_o)

            self.scores_d = tf.nn.xw_plus_b(self.task1_representation, W_d, b_d, name="scores1")
            self.scores_o = tf.nn.xw_plus_b(self.task2_representation, W_o, b_o, name="scores2")


            self.scores_sec_d = list()
            self.scores_sec_o = list()
            #使用mask去掉为空的pair，使其label为[0,0],与空样本的label[0,0]一致
            for i in range(self.sec_disease_num):
                self.scores_sec_d.append(self.mask_sec_d[i]*tf.nn.xw_plus_b(self.task_sec_d_representation[i], W_d, b_d))
            for i in range(self.sec_operation_num): 
                self.scores_sec_o.append(self.mask_sec_o[i]*tf.nn.xw_plus_b(self.task_sec_o_representation[i], W_d, b_d))
          
            self.relation_d = tf.nn.softmax(self.scores_d, name="relation_d")
            self.relation_o = tf.nn.softmax(self.scores_o, name="relation_o")
            self.relation_sec_o = list()
            for i in range(self.sec_operation_num):
                self.relation_sec_o.append(tf.nn.softmax(self.scores_sec_o[i]))


            #output from the graph and session
            self.predictions_d = tf.argmax(self.scores_d, 1, name = "prediction_d") #模型给出的预测结果
            self.predictions_o = tf.argmax(self.scores_o, 1, name = "prediction_o")

            self.predictions_sec_d1 = tf.argmax(self.scores_sec_d[0], 1)
            self.predictions_sec_o1 = tf.argmax(self.scores_sec_o[0], 1)

            #export to tensorflow graph
            #concat??
            self.predictions_sec_d2 = tf.argmax(self.scores_sec_d[1], 1, name="predictions_sec_d2")
            self.y_label_sec_d2 = tf.argmax(self.input_y_sec_d[1], 1, name = 'y_label_sec_d2')
            self.predictions_sec_d3 = tf.argmax(self.scores_sec_d[2], 1, name="predictions_sec_d3")
            self.y_label_sec_d3 = tf.argmax(self.input_y_sec_d[2], 1, name = 'y_label_sec_d3')
            self.predictions_sec_d4 = tf.argmax(self.scores_sec_d[3], 1, name="predictions_sec_d4")
            self.y_label_sec_d4 = tf.argmax(self.input_y_sec_d[3], 1, name = 'y_label_sec_d4')
            self.predictions_sec_o2 = tf.argmax(self.scores_sec_o[1], 1, name="predictions_sec_o2")
            self.y_label_sec_o2 = tf.argmax(self.input_y_sec_o[1], 1, name = 'y_label_sec_o2')
            self.predictions_sec_o3 = tf.argmax(self.scores_sec_o[2], 1, name="predictions_sec_o3")
            self.y_label_sec_o3 = tf.argmax(self.input_y_sec_o[2], 1, name = 'y_label_sec_o3')
            self.predictions_sec_o4 = tf.argmax(self.scores_sec_o[3], 1, name="predictions_sec_o4")
            self.y_label_sec_o4 = tf.argmax(self.input_y_sec_o[3], 1, name = 'y_label_sec_o4')


        # Calculate Mean cross-entropy loss
        with tf.name_scope("loss"):
            #loss的计算方式待修改
            loss_o = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_d, labels=self.input_y_d)
            loss_d = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_o, labels=self.input_y_o)
            loss_o_out = tf.reduce_mean(loss_o)
            loss_d_out = tf.reduce_mean(loss_d)
            losses_sec_d_out_add = tf.constant(0.0)
            losses_sec_o_out_add = tf.constant(0.0)
            losses_sec_d_out = list()
            losses_sec_o_out = list()
            for i in range(self.sec_disease_num):
                loss_sec_d_tmp = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_sec_d[i], labels=self.input_y_sec_d[i])
                current_loss = tf.reduce_mean(loss_sec_d_tmp)
                losses_sec_d_out.append(current_loss)
                losses_sec_d_out_add += current_loss

            for i in range(self.sec_operation_num):
                loss_sec_o_tmp = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_sec_o[i], labels=self.input_y_sec_o[i])
                current_loss = tf.reduce_mean(loss_sec_o_tmp)
                losses_sec_o_out.append(current_loss)
                losses_sec_o_out_add += current_loss

            gap = tf.reduce_sum(tf.square(self.relation_d - self.relation_o), axis=1, keep_dims=True,name='gap')
            constraints_d_o = tf.multiply(self.matrix[self.sec_disease_num], gap)#!!!
            self.constraints_d_o = tf.identity(tf.reduce_mean(constraints_d_o), name="constraints_d_o")

            constraints_os = tf.constant(0.0) #mask?!!!
            for i in range(self.sec_operation_num):
                gap_o = tf.reduce_sum(tf.square(self.relation_d - self.relation_sec_o[i]), axis=1, keep_dims=True)
                constraints_o = tf.multiply(self.matrix[self.sec_disease_num+1+i], gap_o)
                constraints_os += tf.reduce_mean(constraints_o)
            self.constraints_os = tf.identity(constraints_os, name="constraints_os")
            self.loss = loss_o_out + loss_d_out + losses_sec_d_out_add + losses_sec_o_out_add + \
                        self.constraint_lambda * tf.reduce_mean(constraints_d_o)  + \
                        l2_reg_lambda * ( l2_loss_d + l2_loss_o + l2_loss_sec_d + l2_loss_sec_o)


            self.loss = tf.identity(self.loss, name="loss") 

            
            self.loss_sec_d1 = tf.identity(losses_sec_d_out[0], name="loss_sec_d1")
            # self.loss_sec_d2 = tf.identity(losses_sec_d_out[1], name="loss_sec_d2")
            # self.loss_sec_d3 = tf.identity(losses_sec_d_out[2], name="loss_sec_d3")
            # self.loss_sec_d4 = tf.identity(losses_sec_d_out[3], name="loss_sec_d4")
            

            self.loss_sec_o1 = tf.identity(losses_sec_o_out[0], name="loss_sec_o1")
            # self.loss_sec_o2 = tf.identity(losses_sec_o_out[1], name="loss_sec_o2")
            # self.loss_sec_o3 = tf.identity(losses_sec_o_out[2], name="loss_sec_o3")
            # self.loss_sec_o4 = tf.identity(losses_sec_o_out[3], name="loss_sec_o4")
            
            


        #Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions_d = tf.equal(self.predictions_d, tf.argmax(self.input_y_d, 1))
            correct_predictions_o = tf.equal(self.predictions_o, tf.argmax(self.input_y_o, 1))

            self.accuracy_d = tf.reduce_mean(tf.cast(correct_predictions_d, "float"), name="accuracy_d")
            self.accuracy_o = tf.reduce_mean(tf.cast(correct_predictions_o, "float"), name="accuracy_o")

            correct_predictions_sec_d1 = tf.equal(self.predictions_sec_d1, tf.argmax(self.input_y_sec_d[0], 1))
            correct_predictions_sec_o1 = tf.equal(self.predictions_sec_o1, tf.argmax(self.input_y_sec_o[0], 1))

            self.accuracy_sec_d1 = tf.reduce_mean(tf.cast(correct_predictions_sec_d1, "float"), name="accuracy_sec_d1")
            self.accuracy_sec_o1 = tf.reduce_mean(tf.cast(correct_predictions_sec_o1, "float"), name="accuracy_sec_o1")