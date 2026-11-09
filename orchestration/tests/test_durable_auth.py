import copy

import pytest

from orchestration import auth


class AccountStore:
    """Injected repository for testing auth's durable-storage boundary."""
    def __init__(self):
        self.rows = {}

    def create_user(self, row):
        if self.by_email(row['email']):
            raise ValueError('An account with this email already exists')
        self.rows[row['id']] = copy.deepcopy(row)

    def by_email(self, email):
        return next((copy.deepcopy(r) for r in self.rows.values() if r['email'] == email), None)

    def by_id(self, user_id):
        return copy.deepcopy(self.rows.get(user_id))


@pytest.fixture
def store(monkeypatch):
    repository = AccountStore()
    monkeypatch.setattr(auth, 'get_store', lambda: repository)
    monkeypatch.setenv('JWT_SECRET', 'test-only-secret-more-than-thirty-two-characters')
    monkeypatch.setattr(auth, '_get_db_connection', lambda: pytest.fail('Cloud auth must not use local SQLite'))
    return repository


def test_cloud_account_survives_session_recreation_and_checks_password(store):
    user, token = auth.register_user('cloud@example.com', 'correct-password')
    assert 'password_hash' not in user
    restored = auth.get_user_by_token(token)
    assert restored['id'] == user['id']
    logged_in, _ = auth.authenticate_user('cloud@example.com', 'correct-password')
    assert logged_in['id'] == user['id']
    with pytest.raises(ValueError, match='Invalid email or password'):
        auth.authenticate_user('cloud@example.com', 'wrong-password')
    with pytest.raises(ValueError, match='already exists'):
        auth.register_user('CLOUD@example.com', 'another-password')


def test_deleted_cloud_account_is_not_recreated_from_old_token(store):
    user, token = auth.register_user('deleted@example.com', 'correct-password')
    store.rows.clear()
    with pytest.raises(ValueError, match='no longer exists'):
        auth.get_user_by_token(token)


def test_missing_signing_secret_cannot_commit_an_account(store, monkeypatch):
    monkeypatch.delenv('JWT_SECRET')
    with pytest.raises(RuntimeError, match='JWT_SECRET'):
        auth.register_user('unusable@example.com', 'correct-password')
    assert not store.rows
