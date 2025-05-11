WORK_DIR="/mnt/finder/shiyr/code/R1/Search-R1"
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train_refine_sort3_easy_start
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train_refine_score
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train_base_score
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train_refine_overhaul_2_score
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_train_autorefine
template_type="autorefine"

# LOCAL_DIR=$WORK_DIR/data/musique_train_base_score
# template_type="base"

export HF_ENDPOINT=https://hf-mirror.com
mkdir -p $LOCAL_DIR
echo "Data Format: $template_type" >> $LOCAL_DIR/datasource.txt

## process multiple dataset search format train file
DATA=nq,hotpotqa
# DATA=musique
python $WORK_DIR/scripts/data_process/qa_search_train_merge_sort.py --local_dir $LOCAL_DIR --data_sources $DATA --template_type "$template_type"
echo "Train Data: $DATA" >> $LOCAL_DIR/datasource.txt


## process multiple dataset search format test file
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA --template_type "$template_type" --filename "valid_500" --n_subset 500
echo "Valid Data: $DATA" >> $LOCAL_DIR/datasource.txt

DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA --template_type "$template_type" --filename "test"
echo "Test Data: $DATA" >> $LOCAL_DIR/datasource.txt