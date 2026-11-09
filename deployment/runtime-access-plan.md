# Runtime access to finish the Cloud Run release

Project: `project-836457aa-3a05-4a11-8f8`.

Proposed identity: `halluciguard-runtime@project-836457aa-3a05-4a11-8f8.iam.gserviceaccount.com`.

The existing compute identity has Cloud Run administration and image-writing permissions. The application should use a separate identity with only the data access it needs. No downloadable service-account key is proposed.

| Resource | Permission |
| --- | --- |
| Existing secret `HALLUCIGUARD_OPENROUTER_API_KEY` | Secret accessor, on that secret only |
| Existing secret `JWT_SECRET` | Secret accessor, on that secret only |
| Existing Firestore `(default)` database, created for HalluciGuard in `asia-south1` | `roles/datastore.user`, conditional on this exact database resource |

No project owner/editor role, Cloud Run administration, image publishing, other secret access or public database access is proposed for this runtime identity.

These commands are a reviewable plan and have not been executed:

```sh
gcloud iam service-accounts create halluciguard-runtime   --display-name='HalluciGuard application runtime'   --project=project-836457aa-3a05-4a11-8f8

gcloud secrets add-iam-policy-binding HALLUCIGUARD_OPENROUTER_API_KEY   --member='serviceAccount:halluciguard-runtime@project-836457aa-3a05-4a11-8f8.iam.gserviceaccount.com'   --role=roles/secretmanager.secretAccessor --condition=None   --project=project-836457aa-3a05-4a11-8f8

gcloud secrets add-iam-policy-binding JWT_SECRET   --member='serviceAccount:halluciguard-runtime@project-836457aa-3a05-4a11-8f8.iam.gserviceaccount.com'   --role=roles/secretmanager.secretAccessor --condition=None   --project=project-836457aa-3a05-4a11-8f8

gcloud projects add-iam-policy-binding project-836457aa-3a05-4a11-8f8   --member='serviceAccount:halluciguard-runtime@project-836457aa-3a05-4a11-8f8.iam.gserviceaccount.com'   --role=roles/datastore.user   --condition='expression=resource.name=="projects/project-836457aa-3a05-4a11-8f8/databases/(default)",title=halluciguard-database-only'
```

After approval, bind the two existing secret references to the new Cloud Run revision, set `AUTH_STORE=firestore`, and select the proposed identity. Deploy the exact validated image as a no-traffic tagged revision. Run real generation, correction, account and persistence checks, then switch traffic and promote the matching Vercel preview.
