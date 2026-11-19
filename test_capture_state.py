import unittest

from hi import CaptureState, EventConfirmation, zoom_targets


def face(x=100, track_id=None):
    return {"class_name": "face", "track_id": track_id, "box": [x, 80, x + 50, 150]}


class CaptureStateTests(unittest.TestCase):
    def test_all_visible_tracks_can_confirm_in_the_same_three_frames(self):
        state = EventConfirmation()
        heads = [dict(face(x, i), class_name="helmet", confidence=0.9)
                 for i, x in enumerate((100, 300, 500, 700))]
        candidates = [(head, {"rider": None}) for head in heads]
        self.assertEqual(state.update(candidates, 0), [])
        self.assertEqual(state.update(candidates, 1), [])
        self.assertEqual(state.update(candidates, 2), candidates)
        for head, association in candidates:
            state.saved(head, association, 2)
        moved = [(dict(head, box=[x + 250 for x in head["box"]]), association)
                 for head, association in candidates]
        self.assertEqual(state.update(moved, 3), [])

    def test_each_track_requires_consecutive_matching_labels(self):
        state = EventConfirmation()
        head = dict(face(track_id=1), class_name="helmet", confidence=0.9)
        candidate = (head, {"rider": None})
        state.update([candidate], 0)
        state.update([candidate], 1)
        state.update([], 2)
        self.assertEqual(state.update([candidate], 3), [])
        other = (dict(head, class_name="no-helmet"), {"rider": None})
        self.assertEqual(state.update([other], 4), [])
        self.assertEqual(state.update([other], 5), [])
        self.assertEqual(state.update([other], 6), [other])

    def test_different_known_track_at_same_position_is_not_suppressed(self):
        state = EventConfirmation()
        head = dict(face(track_id=1), class_name="helmet", confidence=0.9)
        association = {"rider": None}
        state.saved(head, association, 0)
        other = (dict(head, track_id=2), association)
        state.update([other], 1)
        state.update([other], 2)
        self.assertEqual(state.update([other], 3), [other])

    def test_same_rider_only_counts_once_per_frame_and_cooldown_expires(self):
        state = EventConfirmation()
        rider = dict(face(track_id=4), class_name="rider")
        head = dict(face(track_id=8), class_name="helmet", confidence=0.8)
        close = dict(head, confidence=0.9, track_id=4, zoomed=True)
        candidates = [(head, {"rider": rider}), (close, {"rider": rider})]
        self.assertEqual(state.update(candidates, 0), [])
        self.assertEqual(state.update(candidates, 1), [])
        self.assertEqual(state.update(candidates, 2), [candidates[1]])
        state.saved(close, {"rider": rider}, 2)
        self.assertEqual(state.update(candidates, 3), [])
        self.assertEqual(state.update(candidates, 33), [candidates[1]])

    def test_temporary_loss_of_rider_match_keeps_head_confirmation_and_cooldown(self):
        state = EventConfirmation()
        rider = dict(face(track_id=4), class_name="rider")
        head = dict(face(track_id=8), class_name="helmet", confidence=0.9)
        matched = (head, {"rider": rider})
        unmatched = (head, {"rider": None})
        state.update([matched], 0)
        state.update([unmatched], 1)
        self.assertEqual(state.update([matched], 2), [matched])
        state.saved(head, {"rider": rider}, 2)
        self.assertEqual(state.update([unmatched], 3), [])

    def test_undecided_target_yields_to_people_beyond_first_three(self):
        state = CaptureState()
        people = [face(x) for x in (100, 300, 500, 700)]
        self.assertIs(state.select(people, 0, 0), people[0])
        self.assertIs(state.select(people, 8, 1), people[1])
        self.assertIs(state.select(people, 16, 2), people[2])
        self.assertIs(state.select(people, 24, 3), people[3])
        self.assertIs(state.select(people, 32, 7), people[0])

    def test_jitter_and_changed_track_id_are_suppressed_during_cooldown(self):
        state = CaptureState()
        state.select([face(track_id=1)], 0, 0)
        state.saved(0)
        self.assertIsNone(state.select([face(105, 2)], 1, 1))
        self.assertIsNotNone(state.select([face(105, 2)], 2, 31))

    def test_captured_face_does_not_block_unrelated_rider(self):
        person = face()
        rider = {"class_name": "rider", "track_id": 4, "box": [400, 70, 500, 300]}
        self.assertEqual(zoom_targets([person, rider], [person]), [rider])


if __name__ == "__main__":
    unittest.main()
