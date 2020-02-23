#!/usr/bin/env python
# coding=utf-8

import codecs
import gc
import sys

import numpy as np

from mypreprocess import MyVocabularyProcessor
from util import preprocess_unit, preprocess_arr

reload(sys)
sys.setdefaultencoding("utf-8")


class InputHelper(object):
    def get_data(self, vocab_processor, train_x1, train_x2, train_y, max_document_length):
        """
        Use vocab_processor to index mention and entity pairs and then pad them and return mask arrs
        :param vocab_processor:
        :param train_x1:
        :param train_x2:
        :param train_y:
        :param max_document_length:
        :return:
        """
        train_x1_i = np.asarray(list(vocab_processor.transform(train_x1)))
        train_x2_i = np.asarray(list(vocab_processor.transform(train_x2)))

        mask_train_x1 = np.zeros([len(train_x1_i), max_document_length])
        mask_train_x2 = np.zeros([len(train_x2_i), max_document_length])

        new_mask_x1, new_mask_x2 = self.padding_and_generate_mask(train_x1, train_x2, mask_train_x1, mask_train_x2)
        return (train_x1_i, train_x2_i, new_mask_x1, new_mask_x2, train_y)

    def padding_and_generate_mask(self, x1, x2, new_mask_x1, new_mask_x2):
        """
        Pad the sentence and return mask array for mention and entity pair
        :param x1:
        :param x2:
        :param new_mask_x1:
        :param new_mask_x2:
        :return:
        """

        for i, (x1, x2) in enumerate(zip(x1, x2)):
            # whether to remove sentences with length larger than maxlen
            #if len(x1) == 0 or len(x2) == 0:
            #    print("")
            new_mask_x1[i, 0:len(x1)] = 1.0
            new_mask_x2[i, 0:len(x2)] = 1.0
        return new_mask_x1, new_mask_x2


    def batch_iter(self, all_train_tensor, y_trains,co_arr,data_mask, batch_size, num_epochs, max_record_entity, shuffle=True):
        """
        Generates a batch iterator for a data set.
        :param data:
        :param batch_size:
        :param num_epochs:
        :param shuffle:true!!!!!!!!!!
        :return:
        """
        X = np.asarray(all_train_tensor)
        Y = np.asarray(y_trains)
        coorrence = np.asarray(co_arr)
        Mask = list()
      
        #change 0.0/1.0 to [0.0,0.0] / [1.0,1.0]
        [mask_rows, mask_cols] = data_mask.shape
        for i in range(mask_rows):
            tmp = data_mask[i]
            tmp.tolist()
            line_list = list()
            for j in tmp:
                line_list.append([j])
            Mask.append(line_list)

        for line_index,line in enumerate(Mask):
            for col_index,value in enumerate(line):
                Mask[line_index][col_index] = [value[0],value[0]]

        Mask = np.asarray(Mask)
       
        data_size = len(y_trains[0])
        num_batches_per_epoch = int(data_size / batch_size)

        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                # print "Shuffle!!!!"
                shuffle_indices = np.random.permutation(np.arange(data_size))
                for i in range(max_record_entity):
                    X[i] = X[i][shuffle_indices]
                    Y[i] = Y[i][shuffle_indices]
                    coorrence[i] = coorrence[i][shuffle_indices]
                    Mask[i] = Mask[i][shuffle_indices]

            shuffled_X = X
            shuffled_Y = Y
            shuffled_coorrence = coorrence 
            shuffled_Mask = Mask

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                batch_X = list()
                batch_Y = list()
                batch_Mask = list()
                batch_cor = list()
                for i in range(max_record_entity):
                    batch_X.append(shuffled_X[i][start_index:end_index])
                    batch_Y.append(shuffled_Y[i][start_index:end_index])
                    batch_Mask.append(shuffled_Mask[i][start_index:end_index])
                    batch_cor.append(shuffled_coorrence[i][start_index:end_index])

                yield zip(batch_X,batch_Y,batch_cor,batch_Mask)

    def getTestIndexedDataSet(self, data_path, sep, vocab_processor, max_document_length, y_value):
        """
        Read in labeled test data and use previous vocabulary processor to index them
        :param data_path:
        :param sep:
        :param vocab_processor:
        :param max_document_length:
        :param y_value:
        :return:
        """
        x1_temp, x2_temp, y = self.getTsvTestData(data_path, sep, max_document_length, y_value)

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        return x1, x2, y

    def toVocabularyIndexVector(self, datax1, datax2, vocab_path, max_document_length):
        """
        Transform the word list to vocabulary_index vectors
        :param datax1:
        :param datax2:
        :param vocab_path:
        :param max_document_length:
        :return:
        """
        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        print(len(vocab_processor.vocabulary_))

        datax1 = preprocess_arr(datax1)
        datax2 = preprocess_arr(datax2)
        x1 = np.asarray(list(vocab_processor.transform(datax1)))
        x2 = np.asarray(list(vocab_processor.transform(datax2)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1, x2

    def getTsvData(self, filepath, sep, max_record_entity,sequence_length, y_value=False):
        """
        load the data (label1, mention1, entity1)... (label22, mention22, entity22) from labeled files
        :param filepath:
        :return:  three lists(label_lists, mention_lists, entity_lists
        """

        print("Loading labelled data from " + filepath)
        label_lists = [0]*max_record_entity
        mention_lists = [0]*max_record_entity
        entity_lists = [0]*max_record_entity
        mask_lists = [0]*max_record_entity
        line_num = 0 
        for line in codecs.open(filepath, "r", "utf-8"): 
            line = line.strip().split(sep)
            if len(line) < max_record_entity * 3:
                continue

            #只取要的部分
            items = line[:(max_record_entity/2)*3]
            items.extend(line[11*3:11*3+(max_record_entity/2)*3])
     
            # truncate when length is bigger than the max_length
            for index,item in enumerate(items):
                if index%3 == 0 :
                    content1_fixed = preprocess_unit(item)
                    content2_fixed = preprocess_unit(items[index+1])
                    flag_empty = 0.0 if (content1_fixed == '' and content2_fixed == '') else 1.0
                    if len(content1_fixed) > sequence_length:
                        content1_fixed = content1_fixed[:sequence_length] 
                    if len(content2_fixed)> sequence_length:
                        content2_fixed = content2_fixed[:sequence_length] 
                    if line_num == 0:
                        entity_lists[index/3] = [content1_fixed]
                        mention_lists[index/3] = [content2_fixed]
                        mask_lists[index/3] = [flag_empty]
                        if items[index+2] == '1':
                            label_lists[index/3] = [[1,0]]
                        else:
                            label_lists[index/3] = [[0,1]] if (flag_empty == 1.0) else [[0,0]]
                    else:
                        entity_lists[index/3].append(content1_fixed) #entity,mention list是否需要调换顺序???
                        mention_lists[index/3].append(content2_fixed)
                        mask_lists[index/3].append(flag_empty)
                        if items[index+2] == '1':
                            label_lists[index/3].append([1,0])
                        else:
                            if flag_empty == 1.0:
                                label_lists[index/3].append([0,1]) 
                            else:
                                label_lists[index/3].append([0,0])
                      
            line_num += 1
       
        print('load records %d'%(line_num))
        return np.asarray(mention_lists), np.asarray(entity_lists),np.asarray(label_lists),np.asarray(mask_lists)

    def getTsvTestData_Mul_Labels_Dyna(self, filepath, sep, sequence_length, y_value=False):
        """
        load the data(label, mention, entity) from labeled mutlti-task files
        :param filepath:
        :return:  three lists(label_list, mention_list, entity_list)
        """
        print("Loading testing/labelled data from " + filepath)
        x1, x2, x3, x4 = [], [], [], []
        y = []
        y2 = []
        indicate = []
        for line in codecs.open(filepath, "r", "utf-8"):
            l = line.strip().split(sep)
            l[1] = preprocess_unit(l[1])
            l[2] = preprocess_unit(l[2])
            if len(l[1]) > sequence_length or len(l[2]) > sequence_length:
                l[1] = l[1][:sequence_length]
                l[2] = l[2][:sequence_length]
            x1.append(l[1])
            x2.append(l[2])
            y = self.add_y_helper(y_value, y, int(l[0]) == 1)

            if len(l) == 3:  # dynamic single task1
                x3.append("")
                x4.append("")
                y2 = self.add_y_helper(y_value, y2, False)
                indicate.append(1)
            else:
                l[4] = preprocess_unit(l[4])
                l[5] = preprocess_unit(l[5])
                # truncate when length is bigger than the max_length
                if len(l[4]) > sequence_length or len(l[5]) > sequence_length:
                    l[5] = l[5][:sequence_length]
                    l[4] = l[4][:sequence_length]
                x3.append(l[4])
                x4.append(l[5])
                indicate.append(0)
                y2 = self.add_y_helper(y_value, y2, int(l[3]) == 1)

        return indicate, np.asarray(x1), np.asarray(x2), np.asarray(x3), np.asarray(x4), np.asarray(y), np.asarray(y2)
