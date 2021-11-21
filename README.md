# STFPM
Code for the paper entitled "Student-Teacher Feature Pyramid Matching for Anomaly Detection" (BMVC 2021)


![plot](./figs/arch.jpg)



# Training & testing
Train a model:
```
python main.py train --category carpet
```
After running this command, a directory `snapshots/carpet` should be created.


Evaluate a model:
```
python main.py test --category carpet --checkpoint snapshots/carpet/best.pth.tar
```
This command will evaluate the model specified by --checkpoint argument. 



# Citation

If you find the work useful in your research, please cite our papar.
```
@inproceedings{wang2021student_teacher,
    title={Student-Teacher Feature Pyramid Matching for Anomaly Detection},
    author={Wang, Guodong and Han, Shumin and Ding, Errui and Huang Di},
    booktitle={{The British Machine Vision Conference (BMVC)}},
    year={2021}
}
```
