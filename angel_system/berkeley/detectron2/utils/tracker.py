import os
import numpy as np


class Tracker:
    def __init__(self, metadata, number_frames, last_time, fps):
        self.metadata = metadata.as_dict()
        self.number_frames = number_frames
        self.last_time = last_time
        self.fps = fps
        self.memroy_frame_number = self.last_time * self.fps if self.fps else None
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
        self.step_map = self.metadata['sub_steps']


    def inititae_contact_memory(self):
        contact_pairs_details = self.metadata['contact_pairs_details']

        for i, _step in enumerate(contact_pairs_details):
            sub_list = []
            sub_list.append('step ' + str(i+1))
            sub_list.append(_step)
            for sub_step in _step:
                sub_list.append([sub_step, []])
                #sub_list.append([sub_step, np.zeros([self.number_frames])])
            self.contact_memory.append(sub_list)


    def map_classes(self, tensor):
        name_list = []
        for t in tensor:
            name = self.metadata['thing_classes'][int(t.item() - 1)]
            name_list.append(name)
        return name_list

    def initiate_object_memory(self):
        """

        :return:
        """
        for cate in self.metadata['thing_classes']:
            sub_list = []
            sub_list.append(cate)
            #sub_list.append(np.zeros([self.number_frames]))
            sub_list.append([])
            #sub_list.append(-1 * np.ones([self.number_frames]))
            sub_list.append([])
            #sub_list.append(-1 * np.ones([self.number_frames]))
            sub_list.append([])
            self.object_memory.append(sub_list)




    def update_contact_memory(self, contact_pairs, current_idx):
        #print(self.contact_memory)
        for _contact in contact_pairs:
            for _step in self.contact_memory:
                if _contact in _step[1]:
                    index = _step[1].index(_contact)
                elif [_contact[1], _contact[0]] in _step[1]:
                    index = _step[1].index([_contact[1], _contact[0]])
                else:
                    index = -1
                if index != -1:
                    _step[int(index + 2)][1].append(current_idx - 1)
                    #_step[int(index + 2)][1][current_idx - 1] = 1
                    break

    def update_object_memory(self, obj_list, obj_obj_contact_classes, obj_hand_contact_classes, current_idx):
        #print("obj mem", self.object_memory)
        #import pdb; pdb.set_trace()
        for i, obj in enumerate(obj_list):
            for _cate in self.object_memory:
                if _cate[0] != obj:
                    continue
                _cate[1].append(current_idx - 1)
                _cate[2].append(obj_obj_contact_classes[i])
                _cate[3].append(obj_hand_contact_classes[i])
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
            return 'Need more test !', 'Unknown'
        else:
            current_sub_step = None
            sub_flag = -1
            for _step in self.contact_memory:
                step_name = _step[0]
                for sub_step in _step[2:]:
                    start_frame = current_idx - self.step_last_frame
                    stop_frame = current_idx
                    #flag = sub_step[1][start_frame : stop_frame + 1].sum()
                    flag = len([f for f in sub_step[1] if start_frame <= f <= (stop_frame+1)])
                    # if flag > 2/3 * (self.step_last_frame):
                    if sub_step[0] == ['switch', 'hand'] or sub_step[0] == ['used paper filter', 'trash can']: # Coffee specific
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

            return self.step, self.sub_step, self.next_sub_step, 'Unknown'









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




