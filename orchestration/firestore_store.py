"""Durable accounts/history for Cloud Run, using its existing service identity.

No browser Firestore SDK or service-account key is used. Email uniqueness is
transactional; histories live under the authenticated user's document.
"""
import hashlib
import json
import os
from functools import lru_cache


class FirestoreStore:
    def __init__(self, client=None):
        if client is None:
            from google.cloud import firestore
            client = firestore.Client(project=os.environ.get('GOOGLE_CLOUD_PROJECT'), database=os.environ.get('FIRESTORE_DATABASE', '(default)'))
        self.client = client
        self.users = client.collection('halluciguard_users')
        self.emails = client.collection('halluciguard_email_index')

    def _email_ref(self, email):
        return self.emails.document(hashlib.sha256(email.strip().lower().encode()).hexdigest())

    def by_email(self, email):
        index = self._email_ref(email).get()
        return self.by_id(index.to_dict()['user_id']) if index.exists else None

    def by_id(self, user_id):
        doc = self.users.document(user_id).get()
        return doc.to_dict() if doc.exists else None

    def create_user(self, user):
        from google.cloud import firestore
        email_ref = self._email_ref(user['email'])
        user_ref = self.users.document(user['id'])

        @firestore.transactional
        def create(transaction):
            if email_ref.get(transaction=transaction).exists:
                raise ValueError('An account with this email already exists')
            transaction.create(user_ref, user)
            transaction.create(email_ref, {'user_id': user['id']})

        create(self.client.transaction())

    def update_profile(self, user_id, name, picture):
        fields = {'name': name}
        if picture:
            fields['picture'] = picture
        self.users.document(user_id).update(fields)

    def history(self, user_id):
        return self.users.document(user_id).collection('history')

    def save_history(self, user_id, record, result):
        serialized = json.dumps(result)
        if len(serialized.encode()) > 900_000:
            raise ValueError('Verification exceeds the history document size limit')
        self.history(user_id).document(record['id']).set({**record, 'result_json': serialized})

    def list_history(self, user_id, limit):
        from google.cloud import firestore
        docs = self.history(user_id).order_by('created_at', direction=firestore.Query.DESCENDING).limit(max(1, min(limit, 200))).stream()
        items = []
        for doc in docs:
            item = doc.to_dict()
            item.pop('user_id', None)
            item['result'] = json.loads(item.pop('result_json'))
            items.append(item)
        return items

    def clear_history(self, user_id, history_id=None):
        if history_id:
            self.history(user_id).document(history_id).delete()
        else:
            # Bounded batches remain below Firestore's write batch limit.
            while True:
                docs = list(self.history(user_id).limit(200).stream())
                if not docs:
                    break
                batch = self.client.batch()
                for doc in docs:
                    batch.delete(doc.reference)
                batch.commit()
        return True


@lru_cache(maxsize=1)
def get_store():
    mode = os.environ.get('AUTH_STORE', 'sqlite').strip().lower()
    if mode == 'sqlite':
        return None
    if mode != 'firestore':
        raise RuntimeError('AUTH_STORE must be sqlite or firestore')
    return FirestoreStore()
