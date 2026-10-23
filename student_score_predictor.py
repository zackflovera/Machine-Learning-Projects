import joblib
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

study_hours = [[1], [2], [3], [4], [5], [6], [7], [8]]
exam_scores = [42, 48, 55, 61, 68, 74, 81, 85]

# creates a model based on defined x(study_hours) and y(exam_scores) values
# saves the model as student_score_model.pkl


def find_best_degree(X_data,
                     Y_data,
                     degree_range_end=5):
    best_degree = None
    best_cv_MAE = float("inf")
    for degree in range(1, degree_range_end):
        model = make_pipeline(
            PolynomialFeatures(degree=degree),
            LinearRegression()
        )

        fold_maes = -cross_val_score(
            model,
            X_data,
            Y_data,
            cv=3,
            scoring="neg_mean_absolute_error"
        )
        if fold_maes.mean() < best_cv_MAE:
            best_cv_MAE = fold_maes.mean()
            best_degree = degree

    return best_degree


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    study_hours,
    exam_scores,
    test_size=0.25,
    random_state=42
)

# choose best degree excluding the testing data
best_degree = find_best_degree(X_data=X_train, Y_data=y_train)

# Build the selected model
final_model = make_pipeline(
    PolynomialFeatures(degree=best_degree),
    LinearRegression()
)

# Train using training data only
final_model.fit(X_train, y_train)

# Evaluate on unseen test data
test_predictions = final_model.predict(X_test)
test_mae = mean_absolute_error(y_test, test_predictions)

# save the model
joblib.dump(final_model, "student_score_model.pkl")
print("model saved successfully! ")
