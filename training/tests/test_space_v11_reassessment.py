import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_space_v11 import init
init(os.environ.get('A2UI_V10_POLICY_REPO', str(Path(__file__).resolve().parents[2])))
from ir_training.data.express_preparation import serialize_checked
from space_v11_reassessment import reassess,visible_content_proof


def checked(text):
    import json
    return serialize_checked('<a2ui>\nroot=Column([t])\nt=Text('+json.dumps(text)+')\n</a2ui>','root-first')


class ReassessmentTest(unittest.TestCase):
    def test_regrouped_content(self):
        source='Keep the jar closed.\nDo not open before noon.\nUse 25 mg daily.'
        result=checked('Keep the jar closed. Do not open before noon. Use 25 mg daily.')
        self.assertTrue(visible_content_proof(source,result.graph)['passed'])

    def test_lost_negation(self):
        self.assertFalse(visible_content_proof('Do not open the safety door.',checked('Do open the safety door.').graph)['passed'])

    def test_reordered_numbers(self):
        source='Alpha costs 10 dollars and Beta costs 20 dollars.'
        self.assertFalse(visible_content_proof(source,checked('Alpha costs 20 dollars and Beta costs 10 dollars.').graph)['passed'])

    def test_changed_number(self):
        proof=visible_content_proof('Use 25 mg daily.',checked('Use 250 mg daily.').graph)
        self.assertIn('25',proof['missing_numeric_anchors'])

    def test_unsupported_added_number(self):
        proof=visible_content_proof('Use 25 mg daily.',checked('Use 25 mg daily. The bonus dose is 999 mg.').graph)
        self.assertFalse(proof['passed'])
        self.assertIn('999',proof['unsupported_numeric_anchors'])

    def test_numeric_sign(self):
        proof=visible_content_proof('The outdoor temperature is -5 degrees.',checked('The outdoor temperature is 5 degrees.').graph)
        self.assertFalse(proof['passed'])

    def test_comparator(self):
        proof=visible_content_proof('Choose sizes < 5 cm.',checked('Choose sizes > 5 cm.').graph)
        self.assertFalse(proof['passed'])

    def test_unused_state_does_not_supply_words(self):
        c=checked('Small unrelated answer.')
        c.graph['state']['unused']='The required text must remain visible.'
        self.assertFalse(visible_content_proof('The required text must remain visible.',c.graph)['passed'])

    def test_style_number_is_not_fact(self):
        c=checked('Temperature is mild today.')
        c.graph['elements']['t']['props']['fontSize']=42
        self.assertFalse(visible_content_proof('Temperature is 42 degrees today.',c.graph)['passed'])

    def test_generic_flags_can_clear(self):
        source='Keep the jar closed.\nDo not open before noon.'
        c=checked('Keep the jar closed. Do not open before noon.')
        result=reassess(source,c.text,c,{'eligible':False,'blocking_reasons':[],'review_reasons':['content_unit_fidelity','exact_numbers_dates_units_fbeta']})
        self.assertTrue(result['eligible'])
        self.assertFalse(result['evidence']['rescored'])

    def test_missing_action_never_waived(self):
        source='Keep the jar closed.\nAction: [Button: Read Manual] [URL_1]'
        c=checked('Keep the jar closed.')
        result=reassess(source,c.text,c,{'eligible':False,'blocking_reasons':[],'review_reasons':['missing_or_mismatched_action']})
        self.assertFalse(result['eligible'])
        self.assertTrue(result['evidence']['rescored'])

    def test_saved_reference_map_restores_both_sides(self):
        source='Keep the jar closed.\nAction: [Button: Read Manual] [URL_1]'
        c=serialize_checked('<a2ui>\nroot=Column([t,b])\nt=Text("Keep the jar closed.")\nb=Button("Read Manual",onPress=openUrl("[URL_1]"))\n</a2ui>','root-first')
        result=reassess(source,c.text,c,{'eligible':False,'blocking_reasons':[],'review_reasons':['missing_or_mismatched_action']},{'[URL_1]':{'url':'https://example.org/manual'}})
        self.assertTrue(result['eligible'],result)
        self.assertTrue(result['evidence']['rescored'])

    def test_grouped_number_trailing_punctuation(self):
        proof=visible_content_proof('The price is 850,000, including tax.',checked('The price is 850000, including tax.').graph)
        self.assertTrue(proof['passed'],proof)

    def test_bold_ordered_heading(self):
        proof=visible_content_proof('**2. Check the filter**',checked('Check the filter').graph)
        self.assertTrue(proof['passed'],proof)

    def test_angle_wrapped_reference(self):
        proof=visible_content_proof('Manual reference: <[SOURCE_URL_1]>',checked('Manual reference: [SOURCE_URL_1]').graph)
        self.assertTrue(proof['passed'],proof)

    def test_long_literal_text_is_not_clipped_by_metric(self):
        text=('Pack the supplies carefully.\n'*170)+'Do not forget the emergency water.'
        self.assertGreater(len(text),4096)
        proof=visible_content_proof(text,checked(text).graph)
        self.assertTrue(proof['passed'],proof)

    def test_table_cell_label_bindings(self):
        source='Salary: EUR 48000. Work mode: 3 days office. Skills: Tableau required.'
        c=serialize_checked('<a2ui>\nroot=Table(columns=["Salary","Work mode","Skills"],rows=[["EUR 48000","3 days office","Tableau required"]])\n</a2ui>','root-first')
        proof=visible_content_proof(source,c.graph)
        self.assertTrue(proof['passed'],proof)

    def test_table_swapped_value_bindings_fail(self):
        source='Salary: EUR 48000. Work mode: 3 days office. Skills: Tableau required.'
        c=serialize_checked('<a2ui>\nroot=Table(columns=["Salary","Work mode","Skills"],rows=[["3 days office","EUR 48000","Tableau required"]])\n</a2ui>','root-first')
        self.assertFalse(visible_content_proof(source,c.graph)['passed'])

    def test_arbitrary_table_title_not_used_as_visible(self):
        c=serialize_checked('<a2ui>\nroot=Table(columns=["Name","Value"],rows=[["Test","25"]],title="Missing explanation")\n</a2ui>','root-first')
        self.assertFalse(visible_content_proof('Missing explanation',c.graph)['passed'])

    def test_static_card_title_and_subtitle_are_native_visible(self):
        c=serialize_checked('<a2ui>\nroot=Card([t],title="Day 1: 2027-04-05 Arrival Kyoto",subtitle="Evening walk")\nt=Text("Follow the marked route.")\n</a2ui>','root-first')
        proof=visible_content_proof('Day 1: 2027-04-05 Arrival Kyoto\nEvening walk\nFollow the marked route.',c.graph)
        self.assertTrue(proof['passed'],proof)
        self.assertEqual(proof['native_static_card_label_count'],2)

    def test_email_special_card_title_not_assumed_visible(self):
        c=serialize_checked('<a2ui>\nroot=Card([t],title="Skipped title")\nt=Text("Subject: Hello\\nFrom: Sender\\nTo: Recipient\\nBody: Message")\n</a2ui>','root-first')
        proof=visible_content_proof('Skipped title',c.graph)
        self.assertFalse(proof['passed'],proof)

    def test_dynamic_card_title_not_assumed_visible(self):
        c=serialize_checked('<a2ui>\nroot=Card([t],title="Skipped title")\nt=Text("Other content")\n</a2ui>','root-first')
        c.graph['elements']['root']['props']['visible']=False
        self.assertFalse(visible_content_proof('Skipped title',c.graph)['passed'])

    def test_missing_acceptance_fails(self):
        c=checked('Text')
        self.assertFalse(reassess('Text',c.text,c,None)['eligible'])

    def test_unexplained_ineligibility_fails(self):
        c=checked('Text')
        self.assertFalse(reassess('Text',c.text,c,{'eligible':False})['eligible'])


if __name__=='__main__':
    unittest.main()
