import os
import numpy as np
from .smoothing_tracker_utils import MC50_CATEGORIES as MC50_CATEGORIES

class Tracker:
    def __init__(self, number_frames, last_time, fps):
        self.number_frames = number_frames
        self.last_time = last_time
        self.fps = fps
        self.memroy_frame_number = self.last_time * self.fps
        self.delta_thresh = 5
        self.area_thresh = 10000
        self.edge_thresh = 1/20
        self. step_last_frame = 10

        # this is for initate the contact pair tracker
        self.contact_memory = []
        self.inititae_contact_memory()

        # this is for initate the object tracker
        self.object_memory = []
        self.initiate_object_memory()

        # this is for initiate the initiate the step tracker
        self.step = [-1]
        self.sub_step = [-1]
        # self.next_step = []
        self.next_sub_step = [-1]

        # this is for inititiate the step mapping
        self.initiate_step_mapping()

        self.step_info = []

    def initiate_step_mapping(self):
        sub_steps = {}
        sub_steps['step 1'] = []
        sub_steps['step 2'] = []
        sub_steps['step 3'] = []
        sub_steps['step 4'] = []
        sub_steps['step 5'] = []
        sub_steps['step 6'] = []
        sub_steps['step 7'] = []
        sub_steps['step 8'] = []

        ###################################################step 1###################################################
        ##############################sub-step 1##############################
        sub_step = []
        sub_step.append('Measure 12 ounces of cold water')
        sub_step.append([['measuring cup', 'water']])
        sub_steps['step 1'].append(sub_step)
        del sub_step

        ##############################sub-step 2##############################
        sub_step = []
        sub_step.append('Transfer the water to a kettle')
        sub_step.append([['measuring cup', 'kettle (open)']])
        sub_steps['step 1'].append(sub_step)
        del sub_step

        ##############################sub-step 3##############################
        sub_step = []
        sub_step.append('Turn on the kettle')
        sub_step.append([['switch', 'hand']])
        sub_steps['step 1'].append(sub_step)
        del sub_step

        ###################################################step 2###################################################
        ##############################sub-step 4##############################
        sub_step = []
        sub_step.append('Place the Dripper on top of the mug')
        sub_step.append([['hand', 'filter cone'],
                         ['hand', 'mug'],
                         ['mug', 'filter cone'],
                         ['hand', 'filter cone + mug']])
        sub_steps['step 2'].append(sub_step)
        del sub_step

        ###################################################step 3###################################################
        #############################sub-step 5##############################
        sub_step = []
        sub_step.append('Take the coffee filter and fold it in half into a semicircle.')
        sub_step.append([['paper filter', 'paper filter bag'],
          ['hand', 'paper filter bag'],
          ['hand', 'paper filter'],
          ['hand', 'paper filter (semi)']])
        sub_steps['step 3'].append(sub_step)
        del sub_step

        ##############################sub-step 6##############################
        sub_step = []
        sub_step.append('Fold the coffee filter again to create a quarter circle.')
        sub_step.append([['hand', 'paper filter (quarter)']])
        sub_steps['step 3'].append(sub_step)
        del sub_step

        # #############################sub-step 7##############################
        sub_step = []
        sub_step.append('Place the folder paper into the dripper.')
        sub_step.append([['paper filter (quarter)', 'filter cone + mug']])
        sub_steps['step 3'].append(sub_step)
        del sub_step

        ##############################sub-step 8##############################
        sub_step = []
        sub_step.append('Spread the filter open to create a cone inside the dripper.')
        sub_step.append([['hand', 'paper filter + filter cone + mug']])
        sub_steps['step 3'].append(sub_step)
        del sub_step

        ###################################################step 4###################################################
        ##############################sub-step 9##############################
        sub_step = []
        sub_step.append('Turn on the kitchen scale.')
        sub_step.append([['hand', 'scale (off)'],
                         ['hand', 'scale (on)']])
        sub_steps['step 4'].append(sub_step)
        del sub_step

        ##############################sub-step 9##############################
        sub_step = []
        sub_step.append('Place a bowl on the scale.')
        sub_step.append([['container', 'scale (off)'],
                         ['container', 'scale (on)']])
        sub_steps['step 4'].append(sub_step)
        del sub_step

        ##############################sub-step 10##############################
        sub_step = []
        sub_step.append('Zero out the kitchen scale.')
        sub_step.append([['hand', 'container + scale']])
        sub_steps['step 4'].append(sub_step)
        del sub_step

        ##############################sub-step 11##############################
        sub_step = []
        sub_step.append('Add coffee beans into the bowl until read 25 grams.')
        sub_step.append([['container + scale', 'coffee bag'],
                         ['coffee beans + container + scale', 'coffee bag'],
                         ['hand', 'coffee bag']])
        sub_steps['step 4'].append(sub_step)
        del sub_step

        ##############################sub-step 12##############################
        sub_step = []
        sub_step.append('Poured the measured beans into the coffee grinder.')
        sub_step.append([['coffee beans + container', 'grinder (open)'],
                          ['container', 'grinder (open)']])
        sub_steps['step 4'].append(sub_step)
        del sub_step

        ##############################sub-step 13##############################
        sub_step = []
        sub_step.append('Use timer')
        sub_step.append([['hand', 'timer (else)'],
                         ['hand', 'timer (20)'],
                         ['hand', 'timer (30)']])
        sub_steps['step 4'].append(sub_step)
        del sub_step

        ##############################sub-step 13##############################
        # sub_step = []
        # sub_step.append('unknow')
        # sub_step.append([['x'], ['x']])
        # sub_steps['step 4'].append(sub_step)
        # del sub_step

        # ##############################sub-step 14##############################
        sub_step = []
        sub_step.append('Grind the coffee beans by pressing and holding down the back part')
        sub_step.append([['hand', 'grinder (close)']])
        sub_steps['step 4'].append(sub_step)
        del sub_step

        ##############################sub-step 13##############################
        sub_step = []
        sub_step.append('Pour the grinded coffee beans into the filter cone.')
        sub_step.append([['paper filter + filter cone + mug', 'grinder (open)'],
                         ['paper filter + filter cone', 'grinder (open)'],
                         ['coffee grounds + paper filter + filter cone + mug', 'grinder (open)'],
                         ['coffee grounds + paper filter + filter cone', 'grinder (open)']])
        sub_steps['step 4'].append(sub_step)
        del sub_step

        ###################################################step 5###################################################
        ##############################sub-step 16##############################
        sub_step = []
        sub_step.append('Turn on the thermometer.')
        sub_step.append([['thermometer (open)', 'hand'],
                         ['thermometer (close)', 'hand']])
        sub_steps['step 5'].append(sub_step)
        del sub_step

        ##############################sub-step 17##############################
        sub_step = []
        sub_step.append('Place the end of the thermometer into the water.')
        sub_step.append([['thermometer (open)', 'kettle (open)']])
        sub_steps['step 5'].append(sub_step)
        del sub_step

        ###################################################step 6###################################################
        ##############################sub-step 18##############################
        # sub_step = []
        # sub_step.append('Set timer to 30 seconds.')
        # sub_step.append([['hand', 'timer'],
        #                  ['hand', 'timer (20)'],
        #                  ['hand', 'timer (0)'],
        #                  ['hand', 'timer (30)']])
        # sub_steps['step 6'].append(sub_step)
        # del sub_step

        ##############################sub-step 19##############################
        sub_step = []
        sub_step.append('Pour small amount of water onto the grounds.')
        sub_step.append([['kettle', 'coffee grounds + paper filter + filter cone + mug']])
        sub_steps['step 6'].append(sub_step)
        del sub_step

        ##############################sub-step 20##############################
        # sub_step = []
        # sub_step.append('Wait about 30 seconds.')
        # sub_step.append([
        #     # ['hand', 'timer'],
        #     #              ['hand', 'timer (20)'],
        #     #              ['hand', 'timer (0)'],
        #     #              ['hand', 'timer (30)']
        # ])
        # sub_steps['step 6'].append(sub_step)
        # del sub_step

        ###################################################step 7###################################################
        #############################sub-step 21##############################
        sub_step = []
        sub_step.append(
            'Slowly pour water into the grounds circular motion.')
        sub_step.append([['kettle', 'water + coffee grounds + paper filter + filter cone + mug']])
        sub_steps['step 7'].append(sub_step)
        del sub_step

        ##################################################step 8###################################################
        #############################sub-step 22##############################
        sub_step = []
        sub_step.append('Remove dripper from cup.')
        sub_step.append([['hand', 'used paper filter + filter cone'],
                         ['hand', 'used paper filter + filter cone + mug']])
        sub_steps['step 8'].append(sub_step)

        ##############################sub-step 23##############################
        sub_step = []
        sub_step.append('Remove the coffee grounds and paper filter from the dripper.')
        sub_step.append([
                        # ['hand', 'used paper filter + filter cone'],
                         # ['filter cone', 'used paper filter'],
                         ['hand', 'used paper filter']])
        sub_steps['step 8'].append(sub_step)
        self.step_map = sub_steps

        #############################sub-step 24##############################
        sub_step = []
        sub_step.append('Discard the coffee grounds and paper filter.')
        sub_step.append([['used paper filter', 'trash can']
                         # , ['hand', 'used paper filter']
                         ])
        sub_steps['step 8'].append(sub_step)
        self.step_map = sub_steps


    def inititae_contact_memory(self):
        contact_pairs_details = [[['measuring cup', 'water'],
         ['measuring cup', 'kettle (open)'],
         ['switch', 'hand']],  # step 1

         [['hand', 'filter cone'],
          ['mug', 'filter cone'],
          ['hand', 'filter cone + mug']],  # step 2

         [['paper filter', 'paper filter bag'],
          ['hand', 'paper filter bag'],
          ['hand', 'paper filter'],
          ['hand', 'paper filter (semi)'],
          ['hand', 'paper filter (quarter)'],
          ['paper filter (quarter)', 'filter cone + mug'],
          ['hand', 'paper filter + filter cone + mug']], # step 3

         [['hand', 'scale (on)'],
          ['hand', 'scale (off)'],
          ['hand', 'container + scale'],
          ['scale (on)', 'container'],
          ['scale (off)', 'container'],
          ['container + scale', 'coffee bag'],
          ['coffee beans + container + scale', 'coffee bag'],
          ['coffee beans + container', 'grinder (open)'],
          ['container', 'grinder (open)'],
          ['hand', 'timer (else)'],
          ['hand', 'timer (30)'],
          ['hand', 'timer (20)'],
          ['hand', 'grinder (close)'],
          ['paper filter + filter cone + mug', 'grinder (open)'],
          ['paper filter + filter cone', 'grinder (open)'],
          ['coffee grounds + paper filter + filter cone + mug', 'grinder (open)'],
          ['coffee grounds + paper filter + filter cone', 'grinder (open)']],  # step 4

         [['thermometer (open)', 'kettle (open)'],
          ['thermometer (open)', 'hand'],
         ['thermometer (close)', 'hand']], # step 5

         [['kettle', 'coffee grounds + paper filter + filter cone + mug'],
          # ['kettle', 'water + coffee grounds + paper filter + filter cone + mug'],
          # ['hand', 'timer']
          ], # step 6

        [['kettle', 'water + coffee grounds + paper filter + filter cone + mug']
         # ['kettle', 'used paper filter + filter cone + mug']
         ],  # step 7

         [['mug', 'used paper filter + filter cone'],
          ['hand', 'used paper filter + filter cone'],
          ['hand', 'used paper filter + filter cone + mug'],
          ['used paper filter', 'trash can'],
          ['trash can', 'filter cone'],
          ['hand', 'used paper filter']]  # step 8
         ]

        for i, _step in enumerate(contact_pairs_details):
            sub_list = []
            sub_list.append('step ' + str(i+1))
            sub_list.append(_step)
            for sub_step in _step:
                sub_list.append([sub_step, np.zeros([self.number_frames])])
            self.contact_memory.append(sub_list)


    def map_classes(self, tensor):
        name_list = []
        for t in tensor:
            name = MC50_CATEGORIES[int(t.item() - 1)]['name']
            name_list.append(name)
        return name_list

    def initiate_object_memory(self):
        """

        :return:
        """
        for cate in MC50_CATEGORIES:
            sub_list = []
            sub_list.append(cate['name'])
            sub_list.append(np.zeros([self.number_frames]))
            sub_list.append(-1 * np.ones([self.number_frames]))
            sub_list.append(-1 * np.ones([self.number_frames]))
            self.object_memory.append(sub_list)




    def update_contact_memory(self, contact_pairs, current_idx):
        for _contact in contact_pairs:
            for _step in self.contact_memory:
                if _contact in _step[1]:
                    index = _step[1].index(_contact)
                elif [_contact[1], _contact[0]] in _step[1]:
                    index = _step[1].index([_contact[1], _contact[0]])
                else:
                    index = -1
                if index != -1:
                    _step[int(index + 2)][1][current_idx - 1] = 1
                    break

    def update_object_memory(self, obj_list, obj_obj_contact_classes, obj_hand_contact_classes, current_idx):
        for i, obj in enumerate(obj_list):
            for _cate in self.object_memory:
                if _cate[0] != obj:
                    continue
                _cate[1][current_idx - 1] = 1
                _cate[2][current_idx - 1] = obj_obj_contact_classes[i]
                _cate[3][current_idx - 1] = obj_hand_contact_classes[i]
                break


    def update_memory(self, predictions, current_idx):
        self.current_idx = current_idx
        boxes, labels, obj_obj_contact_classes, obj_obj_contact_scores, obj_hand_contact_classes, obj_hand_contact_scores, Contact_infos, Contact_hand_infos = [i for i in predictions]
        label_list = []
        score_list = []

        for _label in labels:
            percent = _label.split(' ')[-1]
            _class = _label[:-(len(percent) + 1)]
            label_list.append(_class)
            score_list.append(float(percent[:-1]))
        label_list = np.array(label_list)

        self.update_object_memory(label_list, obj_obj_contact_classes, obj_hand_contact_classes, current_idx=current_idx)

        if self.current_idx == 850:
            s = 1

        contact_pairs = []
        for i, _contact in enumerate(Contact_infos):
            if not len(_contact) == 0:
                for j in _contact:
                    if not [label_list[i], label_list[j]] in contact_pairs:
                        contact_pairs.append([label_list[i], label_list[j]])
        for i, _obj in enumerate(label_list):
            hand_obj_contact = obj_hand_contact_classes[i]
            obj_obj_contact = obj_obj_contact_classes[i]
            if hand_obj_contact == 1 and obj_obj_contact == 0:
                contact_pairs.append([_obj, 'hand'])

        # for i, _contact in enumerate(Contact_hand_infos):
        #     if not len(_contact) == 0:
        #         for j in _contact:
        #             if not [label_list[i], label_list[j]] in contact_pairs:
        #                 if label_list[i] == 'hand (left)' or label_list[i] == 'hand (right)':
        #                     label_list[i] = 'hand'
        #                 if label_list[j] == 'hand (left)' or label_list[j] == 'hand (right)':
        #                     label_list[j] = 'hand'
        #                 if label_list[i] == 'timer (0)' or label_list[i] == 'timer (20)' or label_list[i] == 'timer (30)' or label_list[i] == 'timer':
        #                     label_list[i] = 'timer'
        #                 if label_list[j] == 'timer (0)' or label_list[j] == 'timer (20)' or label_list[i] == 'timer (30)' or label_list[i] == 'timer':
        #                     label_list[j] = 'timer'
        #                 if not [label_list[i], label_list[j]] in contact_pairs:
        #                     contact_pairs.append([label_list[i], label_list[j]])


        self.update_contact_memory(contact_pairs, current_idx=current_idx)


    def get_sub_step(self, step_name, sub_name):
        flag_start = -1
        flag_end = -1
        sub_idx = -1
        for i, _sub in enumerate(self.step_map[step_name]):
            if sub_name in _sub[1] or [sub_name[1], step_name[0]] in _sub[1]:
                current_sub_step = _sub[0]
                # find if this sub-step is the first sub-step of this step
                if i == 0:
                    flag_start = 1


                # find if this sub-step is the last ub-step of this step
                if i == len(self.step_map[step_name]):
                    flag_end = 1

                sub_idx = i

                return current_sub_step, flag_start, flag_end, sub_idx
        return None, flag_start, flag_end, sub_idx










    def step_mapping(self, current_idx):
        flag_start =  -1
        flag_end = -1
        sub_idx= -1
        if current_idx < self.step_last_frame:
            # get gt
            file = '/shared/niudt/detectron2/DEMO_Results/2022-11-05/MC_6/all_activities_6.csv'
            import pandas as pd
            ann = pd.read_csv(file, header=None).values
            c = 0
            for x in ann:
                if x[1].split('_')[0] == 'frame':
                    break
                c = c + 1
            ann_list = []
            for i in range(c, ann.shape[0] - 1, 2):
                j = i + 1

                # print(ann[i, 9])
                # print(ann[j, 9])
                if ann[i, 9] != ann[j, 9]:
                    print('error' * 50)
                start_frame = int(ann[i, 2])
                end_frame = int(ann[j, 2])
                ann_list.append([start_frame, end_frame, ann[j, 9]])
            find_flag = 0
            for _ann in ann_list:
                if self.current_idx >= _ann[0] and self.current_idx <= _ann[1]:
                    gt = _ann[2]
                    find_flag = 1
                    break
            if find_flag == 0:
                gt = 'background'
            return 'Need more test !', gt
        else:
            current_sub_step = None
            sub_flag = -1
            for _step in self.contact_memory:
                step_name = _step[0]
                for sub_step in _step[2:]:
                    start_frame = current_idx - self.step_last_frame
                    stop_frame = current_idx
                    flag = sub_step[1][start_frame : stop_frame + 1].sum()
                    # if flag > 2/3 * (self.step_last_frame):
                    if sub_step[0] == ['switch', 'hand'] or sub_step[0] == ['used paper filter', 'trash can']:
                        thresh = 3
                    else:
                        thresh = 6

                    if flag > thresh:
                        current_sub_step, flag_start, flag_end, sub_idx = self.get_sub_step(step_name, sub_step[0])
                        sub_flag = 1
                        break
                if sub_flag == 1:
                    break

            self.update_step_tracker(current_sub_step, flag_start, flag_end, _step[0], sub_idx)

            #get gt
            file = '/shared/niudt/detectron2/DEMO_Results/2022-11-05/MC_6/all_activities_6.csv'
            import pandas as pd
            ann = pd.read_csv(file, header=None).values
            c = 0
            for x in ann:
                if x[1].split('_')[0] == 'frame':
                    break
                c = c + 1
            ann_list = []
            for i in range(c, ann.shape[0] - 1, 2):
                j = i + 1

                # print(ann[i, 9])
                # print(ann[j, 9])
                if ann[i, 9] != ann[j, 9]:
                    print('error' * 50)
                start_frame = int(ann[i, 2])
                end_frame = int(ann[j, 2])
                ann_list.append([start_frame, end_frame, ann[j, 9]])
            find_flag = 0
            for _ann in ann_list:
                if self.current_idx >= _ann[0] and self.current_idx <= _ann[1]:
                    gt = _ann[2]
                    find_flag = 1
                    break
            if find_flag == 0:
                gt = 'background'





            return self.step, self.sub_step, self.next_sub_step, gt









    def update_step_tracker(self, current_sub_step, flag_start, flag_end, _step, sub_idx):
        # update the sub-step list
        # if self.current_idx == 39:
        #     s = 1
        if not current_sub_step == None:
            if self.sub_step[-1] == -1:
                info = {}
                info['sub-step'] = current_sub_step
                info['start_frame'] = self.current_idx
                self.step_last_frame = 60
                self.sub_step.append(info)

                # update the step list
                if self.step[-1] == -1:
                    info_ = {}
                    info_['step'] = _step
                    info_['start_frame'] = self.current_idx
                    self.step.append(info_)
                else:
                    if self.step[-1]['step'] != _step:
                        if not 'end_frame' in list(self.step[-1].keys()):
                            self.step[-1]['end_frame'] = self.current_idx
                        info_ = {}
                        info_['step'] = _step
                        info_['start_frame'] = self.current_idx
                        self.step.append(info_)

                # if flag_start == 1:
                #     info_ = {}
                #     info_['step'] = _step
                #     info_['start_frame'] = self.current_idx
                #     self.sub_step.append(info_)


            else:
                if self.sub_step[-1]['sub-step'] != current_sub_step:
                    if not 'end_frame' in list(self.sub_step[-1].keys()):
                        self.sub_step[-1]['end_frame'] = self.current_idx
                        # self.step_last_frame = 10

                    info = {}
                    info['sub-step'] = current_sub_step
                    info['start_frame'] = self.current_idx
                    self.sub_step.append(info)

                    # update the sub-step list
                    if self.step[-1] == -1:
                        info_ = {}
                        info_['step'] = _step
                        info_['start_frame'] = self.current_idx
                        self.step_last_frame = 60
                        self.step.append(info_)
                    else:
                        if self.step[-1]['step'] != _step:
                            if not 'end_frame' in list(self.step[-1].keys()):
                                self.step[-1]['end_frame'] = self.current_idx
                                # self.step_last_frame = 10
                            info_ = {}
                            info_['step'] = _step
                            info_['start_frame'] = self.current_idx
                            self.step.append(info_)
                else:
                    if 'end_frame' in list(self.sub_step[-1].keys()):
                        if self.current_idx - self.sub_step[-1]['end_frame'] < 200 :
                            del self.sub_step[-1]['end_frame']

        else:
            if self.sub_step[-1] != -1:
                if not 'end_frame' in list(self.sub_step[-1].keys()):
                    self.sub_step[-1]['end_frame'] = self.current_idx
                    self.step_last_frame = 10

                # # update the sub-step list
                # if self.step[-1] == -1:
                #     info_ = {}
                #     info_['step'] = _step
                #     info_['start_frame'] = self.current_idx
                #     self.step.append(info_)
                # else:
                #     if self.step[-1]['step'] != _step:
                #         if not 'end_frame' in list(self.step[-1].keys()):
                #             self.step[-1]['end_frame'] = self.current_idx
                #         info_ = {}
                #         info_['step'] = _step
                #         info_['start_frame'] = self.current_idx
                #         self.step.append(info_)

        # find next sub-step
        if not current_sub_step == None:
            if not sub_idx == len(self.step_map[_step]) - 1:
                next_sub_step = self.step_map[_step][sub_idx + 1][0]
            else:
                if _step != 'step 8':
                    _ = int(_step.split(' ')[1]) + 1
                    next_sub_step = self.step_map['step ' + str(_)][0][0]
                else:
                    next_sub_step = 'end'

            if self.next_sub_step[-1] != -1:
                if self.next_sub_step[-1] != next_sub_step:
                    self.next_sub_step.append(next_sub_step)
            else:
                self.next_sub_step.append(next_sub_step)


        # break
            # else:
            #     print('errors')
        #
        #
        # if not step_name in self.current_step:
        #     self.current_step.append(step_name)
        # if not current_sub_step in self.current_sub_step:
        #     self.current_sub_step.append(current_sub_step)
        # if not next_sub_step in self.next_sub_step:
        #     self.next_sub_step.append(next_sub_step)




class Smoothing:
    def __init__(self):
        s = 1

    def initial_update(self, current_idx, predictions):
        height, weight = predictions._image_size
        boxes_area = predictions.get('pred_boxes').area().detach().numpy()
        boxes_center = predictions.get('pred_boxes').get_centers().detach().numpy()
        classes = predictions.get('pred_classes')
        classes = self.map_classes(tensor=classes)
        obj_obj_contact_classes = predictions.get('obj_obj_contact_classes').detach().numpy()
        obj_hand_contact_classes = predictions.get('obj_hand_contact_classes').detach().numpy()

        current_idx = current_idx - 1

        for _obj in self.tracker:
            cate = _obj[0]
            bbox_center = _obj[1]
            bbox_area = _obj[2]
            obj_contact = _obj[3]
            hand_contact = _obj[4]

            if cate in classes:
                idx = classes.index(cate)
                bbox_center[current_idx] = boxes_center[idx]
                bbox_area[current_idx] = boxes_area[idx]
                obj_contact[current_idx] = obj_obj_contact_classes[idx]
                hand_contact[current_idx] = obj_hand_contact_classes[idx]

        for _obj in self.memory:
            cate = _obj[0]
            bbox_center = _obj[1]
            bbox_area = _obj[2]
            obj_contact = _obj[3]
            hand_contact = _obj[4]

            if cate in classes:
                idx = classes.index(cate)
                bbox_center[current_idx] = boxes_center[idx]
                bbox_area[current_idx] = boxes_area[idx]
                obj_contact[current_idx] = obj_obj_contact_classes[idx]
                hand_contact[current_idx] = obj_hand_contact_classes[idx]

    def get_new_boxes(self, _obj):
        cate = _obj[0]
        bboxes_center = _obj[1]
        bboxes_area = _obj[2]
        obj_contact_classes = _obj[3]
        hand_contact_clsses = _obj[4]

        delta_bboxes_center = bboxes_center[1:] - bboxes_center[: -1]
        delta_bboxes_area = bboxes_area[1:] - bboxes_area[: -1]
        # obj_contact_classes = obj_contact_classes[1:] - obj_contact_classes[: -1]
        # hand_contact_clsses = hand_contact_clsses[1:] - hand_contact_clsses[: -1]


        new_bboxes_area = (bboxes_area[-1] + np.average(delta_bboxes_area[:, 0]))[0]
        new_center = np.array([bboxes_center[-1][0] + np.average(delta_bboxes_center[:, 0]), bboxes_center[-1][1] + np.average(delta_bboxes_center[:, 1])]).tolist()
        # new_center = _obj[3]
        # new_center = bboxes_center[-1] + np.average(delta_bboxes_center)

        return cate, new_center, new_bboxes_area

    def update_preditions(self,
                      boxes_area,
                      boxes_center,
                      classes,
                      obj_obj_contact_classes,
                      obj_hand_contact_classes):

        new_predictions = []


        for _obj in self.memory:
            cate = _obj[0]
            bboxes_center = _obj[1]
            bboxes_area = _obj[2]
            obj_contact_classes = _obj[3]
            hand_contact_clsses = _obj[4]

            # get the average
            a_x= sum(bboxes_center[:, 0]) / len(bboxes_center[:, 0])
            a_y = sum(bboxes_center[:, 1]) / len(bboxes_center[:, 1])
            a_bboxes_area = sum(bboxes_area[:, 0]) / len(bboxes_area[:, 0])
            a_obj_contact_classes = sum(obj_contact_classes[:, 0]) / len(obj_contact_classes[:, 0])
            a_hand_contact_clsses = sum(hand_contact_clsses[:, 0]) / len(hand_contact_clsses[:, 0])

            # get the variation
            v_x = np.var(bboxes_center[:, 0])
            v_y = np.var(bboxes_center[:, 1])
            v_bboxes_area = np.var(bboxes_area[:, 0])
            v_obj_contact_classes = np.var(obj_contact_classes[:, 0])
            v_hand_contact_clsses = np.var(hand_contact_clsses[:,0])

            if cate in classes:
                import math
                idx = classes.index(cate)

                # for modify the error (larger difference compared with the original)
                xy_current = boxes_center[idx]
                # xy_average = [a_x, a_y]

                area_current = boxes_area[idx]
                # area_average = a_bboxes_area
                # delta =  math.sqrt((xy_average[0] - xy_current[0]) ** 2 + (xy_average[1] - xy_current[1]) ** 2)
                # area_difference = abs(area_current - area_average)

                pred_cate, pred_center, pred_area = self.get_new_boxes(_obj)

                delta = abs(math.sqrt((pred_center[0] - xy_current[0]) ** 2 + (pred_center[1] - xy_current[1]) ** 2))
                delta_area = abs(area_current - pred_area)


                if delta > self.delta_thresh: #TODO: TRY TO FIND THE THRSH
                    # supress the wrong predition with larger difference between the current and the memory
                    new_cate, new_center, new_bboxes_area = pred_cate, pred_center, pred_area
                    new_predictions.append([new_cate, new_center, new_bboxes_area, obj_obj_contact_classes[idx], obj_hand_contact_classes[idx]])
                elif delta_area > self.area_thresh:
                    # suppress the one with larger difference of the box area
                    new_predictions.append([cate, boxes_center[idx], pred_area, obj_obj_contact_classes[idx], obj_hand_contact_classes[idx]])
                else:
                    new_predictions.append([cate, boxes_center[idx], obj_obj_contact_classes[idx], obj_hand_contact_classes[idx]])
            else:
                import math

                pred_cate, pred_center, pred_area = self.get_new_boxes(_obj)


                # find if it is in the edge
                pred_x = pred_center[0]
                pred_y = pred_center[0]

                if pred_x == 0 and pred_y == 0:
                    pass

                elif pred_x > (self.weight * (self.edge_thresh)) and pred_x < (self.weight * (1 - self.edge_thresh)):
                    if pred_y > (self.height * (self.edge_thresh)) and pred_y < (self.height * (1 - self.edge_thresh)):
                        pass
                else:
                    new_predictions.append([cate, pred_center, pred_area, _obj[3][-1][0],
                                            _obj[4][-1][0]])
        return new_predictions

    def update(self, predictions, current_idx):

        if current_idx <= self.last_time * self.fps:
            self.initial_update(current_idx, predictions)
        else:
            self.height, self.weight = predictions._image_size
            boxes_area = predictions.get('pred_boxes').area().detach().numpy()
            boxes_center = predictions.get('pred_boxes').get_centers().detach().numpy()
            classes = predictions.get('pred_classes')
            classes = self.map_classes(tensor=classes)
            obj_obj_contact_classes = predictions.get('obj_obj_contact_classes').detach().numpy()
            obj_hand_contact_classes = predictions.get('obj_hand_contact_classes').detach().numpy()

            new_predictions = self.update_preditions(boxes_area, boxes_center, classes, obj_obj_contact_classes, obj_hand_contact_classes)

            self.uodate_memory(new_predictions)
            self.update_tracker(new_predictions)

            new_predictions = self.transfer_format(new_predictions)

            return new_predictions




