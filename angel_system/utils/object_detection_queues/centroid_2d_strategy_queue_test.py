import unittest

from centroid_2d_strategy_queue import Centroid2DStrategyQueue



RESOLUTION_W = 1920
RESOLUTION_H = 1080

class Centroid2DStrategyQueueTest(unittest.TestCase):

    def test_queue_n3_k1_insertion(self):
        """
        Tests proper queue insertion when objects are inserted as strings.
        """
        q = Centroid2DStrategyQueue(
            n=5, center_x=RESOLUTION_W/2, center_y=RESOLUTION_H/2)
        
        # Dog is in the middle of the screen. Mug is in top left of the screen.
        # Computer is near bottom right of screen.
        first_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [1, RESOLUTION_W * 3 // 4, RESOLUTION_W // 2],
            [2, RESOLUTION_W * 3 // 4 + 10, RESOLUTION_W // 2 + 20],
            [1, RESOLUTION_H * 3 // 4, RESOLUTION_H // 2],
            [2, RESOLUTION_H * 3 // 4 + 10, RESOLUTION_H // 2 + 10],
            ['mug', 'computer', 'dog']
        )

        # Ball is in top left of the screen. The butterfly is bottom right
        # of this ball. The cat is in the middle of the screen.
        second_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [1, 2, RESOLUTION_W // 2],
            [2, 4, RESOLUTION_W // 2 + 20],
            [1, 2, RESOLUTION_H // 2],
            [2, 4, RESOLUTION_H // 2 + 10],
            ['ball', 'butterfly', 'cat']
        )

        # Shoes is in bottom right of the screen. The pencil is in the top left
        # of the screen. The child is in the top left of the screen.
        third_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [RESOLUTION_W - 10, 2, 1],
            [RESOLUTION_W, 4, 2],
            [RESOLUTION_H - 10, 2, 1],
            [RESOLUTION_H, 4 , 2],
            ['shoes', 'pencil', 'child']
        )
        q.add(timestamp=1, item=first_objects_detected)
        q.add(timestamp=2, item=second_objects_detected)
        q.add(timestamp=3, item=third_objects_detected)
        
        queue_state = q.get_queue()
        first_timestamped_item, second_timetsamped_item, third_timestamped_item = \
            queue_state[0], queue_state[1], queue_state[2]
        first_top_k, second_top_k, third_top_k = \
            first_timestamped_item[-1], second_timetsamped_item[-1], third_timestamped_item[-1]
        # Recall that each object is a List of Tuples of (centroid distance, detected object)
        self.assertEqual(first_top_k[0][-1], 'dog')
        self.assertEqual(second_top_k[0][-1], 'cat')
        self.assertEqual(third_top_k[0][-1], 'shoes')

    def test_queue_n3_k1_insertion_with_confidence_scores(self):
        """
        Tests proper queue insertion when objects are inserted as Tuples with confidence scores.
        """
        q = Centroid2DStrategyQueue(
            n=5, center_x=RESOLUTION_W/2, center_y=RESOLUTION_H/2)
        
        # Dog is in the middle of the screen. Mug is in top left of the screen.
        # Computer is near bottom right of screen.
        first_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [1, RESOLUTION_W * 3 // 4, RESOLUTION_W // 2],
            [2, RESOLUTION_W * 3 // 4 + 10, RESOLUTION_W // 2 + 20],
            [1, RESOLUTION_H * 3 // 4, RESOLUTION_H // 2],
            [2, RESOLUTION_H * 3 // 4 + 10, RESOLUTION_H // 2 + 10],
            [('mug', 0.1), ('computer', 0.8), ('dog', 0.5)]
        )

        # Ball is in top left of the screen. The butterfly is bottom right
        # of this ball. The cat is in the middle of the screen.
        second_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [1, 2, RESOLUTION_W // 2],
            [2, 4, RESOLUTION_W // 2 + 20],
            [1, 2, RESOLUTION_H // 2],
            [2, 4, RESOLUTION_H // 2 + 10],
            [('ball', 0.9), ('butterfly', 0.3), ('cat', 0.5)]
        )

        # Shoes is in bottom right of the screen. The pencil is in the top left
        # of the screen. The child is in the top left of the screen.
        third_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [RESOLUTION_W - 10, 2, 1],
            [RESOLUTION_W, 4, 2],
            [RESOLUTION_H - 10, 2, 1],
            [RESOLUTION_H, 4 , 2],
            [('shoes', 0.9), ('pencil', 0.3), ('child', 0.5)]
        )
        q.add(timestamp=1, item=first_objects_detected)
        q.add(timestamp=2, item=second_objects_detected)
        q.add(timestamp=3, item=third_objects_detected)
        
        queue_state = q.get_queue()
        first_timestamped_item, second_timetsamped_item, third_timestamped_item = \
            queue_state[0], queue_state[1], queue_state[2]
        first_top_k, second_top_k, third_top_k = \
            first_timestamped_item[-1], second_timetsamped_item[-1], third_timestamped_item[-1]
        # Recall that each object is a List of Tuples:
        # (centroid distance, (detected object, confidence score))
        _, obj_with_conf_score = first_top_k[0]
        self.assertEqual(obj_with_conf_score[0], 'dog')
        _, obj_with_conf_score = second_top_k[0]
        self.assertEqual(obj_with_conf_score[0], 'cat')
        _, obj_with_conf_score = third_top_k[0]
        self.assertEqual(obj_with_conf_score[0], 'shoes')

    def test_queue_n3_k2_insertion(self):
        """
        Tests proper queue insertion when the top 2 objects are inserted as strings.
        """
        q = Centroid2DStrategyQueue(
            n=5, center_x=RESOLUTION_W/2, center_y=RESOLUTION_H/2, k=2)
        
        # Dog is in the middle of the screen. Mug is in top left of the screen.
        # Computer is near bottom right of screen.
        first_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [1, RESOLUTION_W * 3 // 4, RESOLUTION_W // 2],
            [2, RESOLUTION_W * 3 // 4 + 10, RESOLUTION_W // 2 + 20],
            [1, RESOLUTION_H * 3 // 4, RESOLUTION_H // 2],
            [2, RESOLUTION_H * 3 // 4 + 10, RESOLUTION_H // 2 + 10],
            ['mug', 'computer', 'dog']
        )

        # Ball is in top left of the screen. The butterfly is bottom right
        # of this ball. The cat is in the middle of the screen.
        second_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [1, 2, RESOLUTION_W // 2],
            [2, 4, RESOLUTION_W // 2 + 20],
            [1, 2, RESOLUTION_H // 2],
            [2, 4, RESOLUTION_H // 2 + 10],
            ['ball', 'butterfly', 'cat']
        )

        # Shoes is in bottom right of the screen. The pencil is in the top left
        # of the screen. The child is in the top left of the screen.
        third_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [RESOLUTION_W - 10, 2, 1],
            [RESOLUTION_W, 4, 2],
            [RESOLUTION_H - 10, 2, 1],
            [RESOLUTION_H, 4 , 2],
            ['shoes', 'pencil', 'child']
        )
        q.add(timestamp=1, item=first_objects_detected)
        q.add(timestamp=2, item=second_objects_detected)
        q.add(timestamp=3, item=third_objects_detected)
        
        queue_state = q.get_queue()
        first_timestamped_item, second_timetsamped_item, third_timestamped_item = \
            queue_state[0], queue_state[1], queue_state[2]
        first_top_k, second_top_k, third_top_k = \
            first_timestamped_item[-1], second_timetsamped_item[-1], third_timestamped_item[-1]
        # Recall that each object is a List of Tuples of (centroid distance, detected object)
        
        first_object_labels = [label for centroid, label in first_top_k]
        self.assertEqual(["dog", "computer"], first_object_labels)

        second_object_labels = [label for centroid, label in second_top_k]
        self.assertEqual(["cat", "butterfly"], second_object_labels)

        third_object_labels = [label for centroid, label in third_top_k]
        self.assertEqual(["shoes", "pencil"], third_object_labels)

    def test_queue_n3_k2_removal(self):
        """
        Tests proper queueing of the last 3 top 2 objects are inserted as strings.
        """
        q = Centroid2DStrategyQueue(
            n=5, center_x=RESOLUTION_W/2, center_y=RESOLUTION_H/2, k=2)
        
        # Dog is in the middle of the screen. Mug is in top left of the screen.
        # Computer is near bottom right of screen.
        first_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [1, RESOLUTION_W * 3 // 4, RESOLUTION_W // 2],
            [2, RESOLUTION_W * 3 // 4 + 10, RESOLUTION_W // 2 + 20],
            [1, RESOLUTION_H * 3 // 4, RESOLUTION_H // 2],
            [2, RESOLUTION_H * 3 // 4 + 10, RESOLUTION_H // 2 + 10],
            ['mug', 'computer', 'dog']
        )

        # Ball is in top left of the screen. The butterfly is bottom right
        # of this ball. The cat is in the middle of the screen.
        second_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [1, 2, RESOLUTION_W // 2],
            [2, 4, RESOLUTION_W // 2 + 20],
            [1, 2, RESOLUTION_H // 2],
            [2, 4, RESOLUTION_H // 2 + 10],
            ['ball', 'butterfly', 'cat']
        )

        # Shoes is in bottom right of the screen. The pencil is in the top left
        # of the screen. The child is in the top left of the screen.
        third_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [RESOLUTION_W - 10, 2, 1],
            [RESOLUTION_W, 4, 2],
            [RESOLUTION_H - 10, 2, 1],
            [RESOLUTION_H, 4 , 2],
            ['shoes', 'pencil', 'child']
        )
        q.add(timestamp=1, item=first_objects_detected)
        q.add(timestamp=2, item=second_objects_detected)
        q.add(timestamp=3, item=third_objects_detected)
        
        no_items = q.get_n_before(timestamp=1)
        self.assertEqual([], no_items)
        last_n_top_k = q.get_n_before(timestamp=4)
        discarded_timestamp, first_top_k_with_centroid_dist = last_n_top_k[0]
        first_top_k = [item for discarded_dist, item in first_top_k_with_centroid_dist]
        self.assertEqual(["dog", "computer"], first_top_k)

        discarded_timestamp, second_top_k_with_centroid_dist = last_n_top_k[1]
        second_top_k = [item for dist, item in second_top_k_with_centroid_dist]
        self.assertEqual(["cat", "butterfly"], second_top_k)

        discarded_timestamp, third_top_k_with_centroid_dist = last_n_top_k[2]
        third_top_k = [item for dist, item in third_top_k_with_centroid_dist]
        self.assertEqual(["shoes", "pencil"], third_top_k)

    def test_queue_n2_k2_removal_with_confidence_scores(self):
        """
        Tests proper queue removal of the last 2 top 2 detected objects before a given timestamp
        when the top 2 objects are inserted as strings with confidence scores.
        """
        q = Centroid2DStrategyQueue(
            n=2, center_x=RESOLUTION_W/2, center_y=RESOLUTION_H/2, k=2)
        
        # Dog is in the middle of the screen. Mug is in top left of the screen.
        # Computer is near bottom right of screen.
        first_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [1, RESOLUTION_W * 3 // 4, RESOLUTION_W // 2],
            [2, RESOLUTION_W * 3 // 4 + 10, RESOLUTION_W // 2 + 20],
            [1, RESOLUTION_H * 3 // 4, RESOLUTION_H // 2],
            [2, RESOLUTION_H * 3 // 4 + 10, RESOLUTION_H // 2 + 10],
            [('mug', 0.1), ('computer', 0.8), ('dog', 0.5)]
        )

        # Ball is in top left of the screen. The butterfly is bottom right
        # of this ball. The cat is in the middle of the screen.
        second_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [1, 2, RESOLUTION_W // 2],
            [2, 4, RESOLUTION_W // 2 + 20],
            [1, 2, RESOLUTION_H // 2],
            [2, 4, RESOLUTION_H // 2 + 10],
            [('ball', 0.9), ('butterfly', 0.3), ('cat', 0.5)]
        )

        # Shoes is in bottom right of the screen. The pencil is in the top left
        # of the screen. The child is in the top left of the screen.
        third_objects_detected = Centroid2DStrategyQueue.BoundingBoxes(
            [RESOLUTION_W - 10, 2, 1],
            [RESOLUTION_W, 4, 2],
            [RESOLUTION_H - 10, 2, 1],
            [RESOLUTION_H, 4 , 2],
            [('shoes', 0.9), ('pencil', 0.3), ('child', 0.5)]
        )
        q.add(timestamp=1, item=first_objects_detected)
        q.add(timestamp=2, item=second_objects_detected)
        q.add(timestamp=3, item=third_objects_detected)
        
        no_items = q.get_n_before(timestamp=1)
        self.assertEqual([], no_items)
        last_n_top_k = q.get_n_before(timestamp=4)
        discarded_timestamp, first_top_k_with_centroid_dist = last_n_top_k[0]
        first_scored_top_k = [scored_item for discarded_dist, scored_item in
                       first_top_k_with_centroid_dist]
        first_top_k = [item for item, score in first_scored_top_k]
        self.assertEqual(["cat", "butterfly"], first_top_k)

        discarded_timestamp, second_top_k_with_centroid_dist = last_n_top_k[1]
        second_scored_top_k = [scored_item for discarded_dist, scored_item in
                        second_top_k_with_centroid_dist]
        second_top_k = [item for item, score in second_scored_top_k]
        self.assertEqual(["shoes", "pencil"], second_top_k)

if __name__ == "__main__":
    unittest.main()