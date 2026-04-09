from isomorphic.utils.logger import ResultsLogger

def test_results_logger(tmp_path):
    logger = ResultsLogger(tmp_path)
    assert logger.log_experiment("exp_test", {"ok": True}).exists()
    assert logger.write_csv("t.csv", [{"a": 1}]).exists()
