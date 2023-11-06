[1mdiff --cc ros/angel_system_nodes/angel_system_nodes/visual_question_answerer.py[m
[1mindex 22b27e84,e7df1e98..00000000[m
[1m--- a/ros/angel_system_nodes/angel_system_nodes/visual_question_answerer.py[m
[1m+++ b/ros/angel_system_nodes/angel_system_nodes/visual_question_answerer.py[m
[36m@@@ -395,11 -404,9 +404,11 @@@[m [mclass VisualQuestionAnswerer(BaseDialog[m
                      obj, score = obj_score[m
                      observables.add(obj)[m
              observables = observables - self.object_dtctn_ignorables[m
[31m-             return ", ".join(observables)[m
[31m -        return observables [m
[32m++            return observables[m
[32m +        else: [m
[32m +            return "nothing"[m
  [m
[31m-     def _get_latest_observables(self, curr_time: int, n: int) -> str:[m
[32m+     def _get_latest_observables(self, curr_time: int, n: int) -> Set:[m
          """[m
          Returns a comma-delimited list of all observed objects per all[m
          entities in self.detected_objects_queue that occurred before a provided time.[m
[36m@@@ -420,9 -427,7 +429,9 @@@[m
              for obj in detection.entity:[m
                  observables.add(obj)[m
          observables = observables - self.object_dtctn_ignorables[m
[32m +        if len(observables)==0:[m
[32m +            return "nothing"[m
[31m-         return ", ".join(observables)[m
[32m+         return observables[m
  [m
      def get_response([m
          self,[m
[36m@@@ -479,17 -476,22 +480,28 @@@[m
          This is the main ROS node listener callback loop that will process[m
          all messages received via subscribed topics.[m
          """[m
[31m-         self.log.info(f"Received message:\n\n{msg.utterance_text}")[m
[32m+         self.log.info(f"Received message: \"{msg.utterance_text}\"")[m
          if not self._apply_filter(msg):[m
              return[m
[32m +        [m
[32m +        msg.utterance_text= msg.utterance_text.replace("Angel, ", "")[m
[32m +        msg.utterance_text= msg.utterance_text.replace("angel, ", "")[m
[32m +        msg.utterance_text= msg.utterance_text.replace("angel", "")[m
[32m +        msg.utterance_text= msg.utterance_text.replace("Angel", "")[m
[32m +        msg.utterance_text= msg.utterance_text.capitalize()[m
          self.question_queue.put(msg)[m
  [m
[32m+     def _get_optional_fields_string(self, emotion: str, current_step: str,[m
[32m+                                     current_action: str) -> str:[m
[32m+         optional_fields_string = "\n"[m
[32m+         if emotion:[m
[32m+             optional_fields_string += f"Emotion: {emotion}\n"[m
[32m+         if current_step:[m
[32m+             optional_fields_string += f"My Current Step: {current_step}\n"[m
[32m+         if current_action:[m
[32m+             optional_fields_string += f"My Current Action: {current_action}\n"[m
[32m+         return optional_fields_string.rstrip("\n")[m
[32m+ [m
      def process_question_queue(self):[m
          """[m
          Constant loop to process received questions.[m
[36m@@@ -498,33 -500,44 +510,44 @@@[m
          while True:[m
              question_msg = self.question_queue.get()[m
              start_time = self._get_sec(question_msg)[m
[31m-             self.log.info(f"Processing utterance {question_msg.utterance_text}")[m
[31m- [m
[31m-             # Get most recently detected action.[m
[31m-             action = self._get_latest_action(start_time)[m
[31m-             self.log.info(f"Latest action: {action}")[m
[32m+             self.log.info(f"Processing utterance \"{question_msg.utterance_text}\"")[m
  [m
[32m+             # Get the optional fields.[m
[32m+             optional_fields = \[m
[32m+                 self._get_optional_fields_string(question_msg.emotion, self._get_current_step(),[m
[32m+                                                  self._get_latest_action(start_time))[m
              # Get centered detected objects.[m
              centered_observables = self._get_latest_centered_observables(start_time)[m
[31m-             self.log.info(f"Observed objects: {centered_observables}")[m
[31m- [m
              # Get all detected objects.[m
[31m-             all_observables = self._get_latest_observables([m
[31m-                 start_time, self.object_dtctn_last_n_obj_detections[m
[31m-             )[m
[31m-             self.log.info(f"Observed objects: {all_observables}")[m
[31m- [m
[31m-             # Generate response.[m
[31m-             response = self.get_response([m
[31m-                 question_msg,[m
[31m-                 self._get_dialogue_history(),[m
[31m-                 self._get_current_step(),[m
[31m-                 action,[m
[31m-                 centered_observables,[m
[31m-                 all_observables,[m
[31m-             )[m
[32m+             all_observables = \[m
[32m+                 self._get_latest_observables(start_time, self.object_dtctn_last_n_obj_detections)[m
[32m+ [m
[32m+             response = None[m
[32m+             is_object_clarification = \[m
[32m+                 question_msg.intent and question_msg.intent == INTENT_LABELS[3][m
[32m+             if is_object_clarification and len(centered_observables) > 1:[m
[32m+                 # Object Clarification override: If an associated intent exists and indicates[m
[32m+                 # object clarification in the presence of multiple objects, override the response with[m
[32m+                 # a clarification question.[m
[32m+                 self.log.info([m
[32m+                     "Received confusing object clarification question from user " +\[m
[32m+                         f"about multiple objects: ({centered_observables}). " +\[m
[32m+                         "Inquiring for more details...")[m
[32m+                 response = "It seems you are asking about an object you are unsure about. " +\[m
[32m+                     "I am detecting the following: {}. ".format(centered_observables) +\[m
[32m+                     "Is the object you are referenceing one of these objects?"[m
[32m+             else:[m
[32m+                 all_observables -= centered_observables[m
[32m+                 # Normal response generation.[m
[32m+                 response = self.get_response([m
[32m+                     question_msg,[m
[32m+                     self._get_dialogue_history(),[m
[32m+                     ", ".join(centered_observables) if centered_observables else "Nothing",[m
[32m+                     ", ".join(all_observables) if all_observables else "Nothing",[m
[32m+                     optional_fields[m
[32m+                 )[m
              self.publish_generated_response(question_msg.utterance_text, response)[m
[31m -            self._add_dialogue_history(question_msg.utterance_text, response)[m
[32m +            self._add_dialogue_history(question_msg.utterance_text, response,self.get_emotion_or(question_msg))[m
  [m
      def publish_generated_response(self, utterance: str, response: str):[m
          msg = SystemTextResponse()[m
[1mdiff --cc ros/angel_system_nodes/configs/llm_prompts/vis_qa_teacher_prompt[m
[1mindex c82adf91,54cf8bf9..00000000[m
[1m--- a/ros/angel_system_nodes/configs/llm_prompts/vis_qa_teacher_prompt[m
[1m+++ b/ros/angel_system_nodes/configs/llm_prompts/vis_qa_teacher_prompt[m
[36m@@@ -1,14 -1,13 +1,14 @@@[m
[31m- You are a professional chef teaching me how to best make this recipe. I will ask you a question about cooking and you should respond with a short and efficient answer. To provide an answer, use the context below and if you do not have an answer, say "Sorry I can't help you with that". [m
[31m -You are a teacher helping me learn how to complete a Task. I will tell you how I am feeling (positive, negative, neutral), all the objects that I can see, and what I am currently doing. I will ask you a question and you will respond with an answer.[m
[32m++You are a professional chef teaching me how to best make this recipe. I will ask you questions about cooking and you should respond with a short and efficient answer. To provide an answer, use the context below and if you do not have an answer, say "Sorry I can't help you with that". [m
  [m
[31m -Task Steps:[m
[32m +Here is some context:[m
[32m +Currently I am working on the following recipe:[m
  {recipe}[m
  [m
[31m -{optional_fields}[m
[31m -Objects In Front of Me: {centered_observables}[m
[31m -Objects Nearby: {all_observables}[m
[32m +I finished all steps up until but not including: {current_step} {action}[m
  [m
[31m -Chat: [m
[32m +Here are objects that I see: {centered_observables}, and objects that you can see: {all_observables}[m
[32m +[m
[32m +Our conversation so far:[m
  {chat_history}[m
[31m -Me: {question}[m
[31m -You:[m
[32m +Me ({emotion}): {question}[m
[32m +Your Answer (short, helpful with empathy):[m
