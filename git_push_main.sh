git add .

if [ "$1" = "" ]
then
    git commit -m "Update"
else
    git commit -m "$1"
fi

git push origin main