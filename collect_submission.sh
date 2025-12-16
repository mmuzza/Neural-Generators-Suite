rm -f assignment_submission.zip
zip -r assignment_submission.zip configs models/*py losses/*py utils/*.py  outputs/vae/*.pth outputs/gan/*.pth outputs/diffusion/*.pth *.py

