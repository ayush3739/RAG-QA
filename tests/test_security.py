from datetime import timedelta

import pytest

from backend.services.security import (
    create_access_token,
    decode_token,
    validate_password,
)


def test_created_access_token_can_be_decoded():
    assert decode_token(create_access_token(subject=42)) == 42


def test_expired_access_token_is_rejected():
    expired_token = create_access_token(subject=42, expires_delta=timedelta(seconds=-1))
    assert decode_token(expired_token) is None


def test_password_requires_minimum_length():
    with pytest.raises(ValueError, match="at least"):
        validate_password("too-short")


def test_password_rejects_bcrypt_overflow():
    with pytest.raises(ValueError, match="72 bytes"):
        validate_password("a" * 73)
