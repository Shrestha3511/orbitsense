import math

from uncertainty_model import combined_sigma, probability_of_collision, sigma_model


def test_probability_of_collision_decreases_with_miss_distance():
    sigma_m = 1000.0
    hard_body_m = 10.0
    pc_close = probability_of_collision(500.0, sigma_m, hard_body_m)
    pc_far = probability_of_collision(5000.0, sigma_m, hard_body_m)
    assert pc_close > pc_far


def test_sigma_model_and_combined_sigma_behavior():
    sigma_1h = sigma_model(1.0)
    sigma_4h = sigma_model(4.0)
    assert sigma_4h > sigma_1h
    assert math.isclose(combined_sigma(2.0, 3.0), math.sqrt(sigma_model(2.0) ** 2 + sigma_model(3.0) ** 2))
