import unittest

from routing_engine import (RoutingWeights, add_routing_costs, attach_predicted_risk, create_topology,
                            path_metrics, route, set_failures, train_failure_predictor)


class RoutingEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _ = train_failure_predictor()

    def prepared_graph(self):
        graph = create_topology()
        attach_predicted_risk(graph, self.model)
        add_routing_costs(graph, RoutingWeights())
        return graph

    def test_both_strategies_find_a_route(self):
        graph = self.prepared_graph()
        self.assertIsNotNone(route(graph, "D1", "GW", "baseline"))
        self.assertIsNotNone(route(graph, "D1", "GW", "intelligent"))

    def test_failed_link_can_trigger_reroute(self):
        graph = self.prepared_graph()
        before = route(graph, "D1", "GW", "baseline")
        set_failures(graph, [], [(before[0], before[1])])
        after = route(graph, "D1", "GW", "baseline")
        self.assertIsNotNone(after)
        self.assertNotEqual(before, after)

    def test_metrics_are_bounded(self):
        graph = self.prepared_graph()
        metrics = path_metrics(graph, route(graph, "D1", "GW", "intelligent"))
        self.assertEqual(metrics["status"], "Reachable")
        self.assertGreater(metrics["latency_ms"], 0)
        self.assertGreaterEqual(metrics["packet_delivery_pct"], 0)
        self.assertLessEqual(metrics["packet_delivery_pct"], 100)


if __name__ == "__main__":
    unittest.main()
