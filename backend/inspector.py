import pickle
import pandas as pd
import os

artifacts_folder = "inference_artifacts"
data_folder = "../Data"

def find_different_user():
    print("\n--- 1. Loading Full Data ---")
    try:
        # parquet 파일들을 로드합니다.
        parquet_files = [os.path.join(data_folder, f"{i:012d}.parquet") for i in range(4)]
        columns_to_read = ['session_id', 'c1_name']
        df = pd.read_parquet(parquet_files, columns=columns_to_read)
        
        # 카테고리 이름이 없는(null) 데이터는 제외합니다.
        df.dropna(subset=['c1_name'], inplace=True)
        print("Data loaded and cleaned.")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 각 사용자의 최다 관심 카테고리를 찾습니다.
    session_top_category = df.groupby('session_id')['c1_name'].agg(lambda x: x.value_counts().index[0])

    # '미술 및 공예' 관련 카테고리를 제외합니다.
    arts_cats = ['Other Arts & Crafts', 'Arts & Crafts', 'Drawing Supplies']
    non_arts_sessions = session_top_category[~session_top_category.isin(arts_cats)]

    if non_arts_sessions.empty:
        print("Could not find a user with non-arts interests.")
        return

    # 조건을 만족하는 사용자 중 한 명을 선택합니다.
    target_session_id = non_arts_sessions.index[1] # 2번째 사용자를 선택하여 이전과 다른 결과를 유도

    print(f"\n--- New User Found ---")
    print(f"Session ID: {target_session_id}")
    
    # 해당 사용자의 상위 관심사를 요약하여 보여줍니다.
    user_history_df = df[df['session_id'] == target_session_id]
    top_categories = user_history_df['c1_name'].value_counts().nlargest(3)

    print("\n[User's Preference Summary]")
    print("Top 3 Categories:")
    for cat, count in top_categories.items():
        print(f"- {cat} ({count} times)")


if __name__ == '__main__':
    find_different_user() 