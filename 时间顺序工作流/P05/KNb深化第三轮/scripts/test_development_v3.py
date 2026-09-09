import unittest

from development_v3 import configurations


class DevelopmentV3Tests(unittest.TestCase):
    def test_single_module_stages_keep_other_modules_off(self):
        n_configs = configurations("N")
        self.assertEqual(n_configs[0], ("000_fixed", (False, False, False), "fixed"))
        self.assertTrue(all(config[1][0] is False and config[1][2] is False for config in n_configs))
        k_configs = configurations("K")
        self.assertTrue(all(config[1][1] is False and config[1][2] is False for config in k_configs))
        b_configs = configurations("b")
        self.assertTrue(all(config[1][0] is False and config[1][1] is False for config in b_configs))

    def test_interaction_has_all_eight_module_combinations(self):
        configs = configurations("interaction")
        self.assertEqual(len(configs), 8)
        self.assertEqual({config[1] for config in configs}, {
            (False, False, False), (False, False, True), (False, True, False), (False, True, True),
            (True, False, False), (True, False, True), (True, True, False), (True, True, True),
        })


if __name__ == "__main__":
    unittest.main()
