    str01=""
count=0
for i in "$@";do
    if [ $count == '0' ]
    then
        str01+="$i"
        # echo ${i}
        # echo $str01
    else
        str01+=" ""$i"
    fi 
    count=`expr 1 + $count`
done
# echo $str01
git add * 
git commit -m "$str01"
git push