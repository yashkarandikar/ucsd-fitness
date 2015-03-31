#include<string>
#include<fstream>
#include<iostream>
#include<stdlib.h>
#include<sstream>
#include<algorithm>
#include<random>
#include<chrono>

using namespace std;

double** alloc_matrix(int N, int M);

class Matrix
{
    public:
        int shape[2];
        double **data;

        Matrix(int nrows, int ncols)
        {
            data = alloc_matrix(nrows, ncols);
            shape[0] = nrows;
            shape[1] = ncols;
        }

        double* operator[](int i)
        {
            return data[i];
        }
};

double** alloc_matrix(int N, int M)
{
    double **mat = new double*[N];
    for (int i = 0; i < N; i++)
        mat[i] = new double[M];
    return mat;
}

double** zeros(int nrows, int ncols)
{
    double **z = zeros(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0 ; j < ncols; j++) {
            z[i][j] = 0;
        }
    }
    return z;
}

double **read_matrix(const char *infile, int& N)
{
    // count lines
    N = 0;
    ifstream f;
    f.open(infile);
    string line;
    while (std::getline(f, line))
        N++;
    cout << "N = " << N << "\n";
    f.close();

    f.open(infile);
    double **data = alloc_matrix(N, 3);
    double uid, pid;
    int i = 0;
    double r;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        if (!(iss >> uid >> pid >> r)) 
             break;
        data[i][0] = uid;
        data[i][1] = pid;
        data[i][2] = r;
        i++;
    }
    f.close();
    return data;
}

void init_random(double **a, int n, int m)
{
    for (int i = 0 ; i < n; i++)
        for (int j = 0 ; j < m; j++)
            a[i][j] = ((double)rand()/(double)RAND_MAX);
}

void init_random(vector<double>& a)
{
    int n = a.size();
    for (int i = 0; i < n; i++)
        a[i] = ((double)rand()/(double)RAND_MAX);
}

void clear(double *a, int n)
{
    for (int i = 0; i < n; i++)
        a[i] = 0;
}

void clear(vector<double>& v)
{
    clear(&v[0], v.size());
}

int randint(int low, int high)
{
    int range = high - low + 1;
    return (low + (rand() % range));
}

double get_alpha_ue(vector<double>& theta, int u, int e, int E, int& index)
{
    // theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    index = u * E + e;
    return theta[index];
}

double get_alpha_e(vector<double>& theta, int e, int E, int U, int& index) 
{
    // theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    index = U * E + e;
    return theta[index]; 
}

double get_theta_0(vector<double>& theta)
{
    return theta[theta.size() - 2];
}

double get_theta_1(vector<double>& theta)
{
    return theta[theta.size() - 1];
}

int get_user_count(Matrix& data) 
{
    int nrows = data.shape[0];
    return int(data[nrows-1][0] + 1);
}

vector<int> find_best_path_DP(Matrix& M, double& leastError)
{
    int E = M.shape[0];
    int Nu = M.shape[1];
    //print "Size of M matrix : ", M.shape

    // base case
    double **D = zeros(E, Nu);
    double **decision = zeros(E, Nu);
    for (int i = 0; i < E; i++) {
        D[i][0] = M[i][0];
    }

    // fill up remaining matrix
    double o1, o2;
    for (int n = 1; n < Nu; n++) {
        for (int m = 0 ; m < E; m++) {
            o1 = numeric_limits<double>::max();
            if (m > 0)
                o1 = D[m-1][n-1];
            o2 = D[m][n-1];
            if (o1 < o2) {
                D[m][n] = M[m][n] + o1;
                decision[m][n] = m - 1;
            }
            else {
                D[m][n] = M[m][n] + o2;
                decision[m][n] = m;
            }
        }
    }

    // trace path
    leastError = numeric_limits<double>::max();
    int bestExp = 0;
    // first compute for last workout
    for (int i = 0; i < E; i++) {
        if (D[i][Nu - 1] < leastError) {
            leastError = D[i][Nu-1];
            bestExp = i;
        }
    }
    vector<int> path(Nu);
    path[Nu - 1] = bestExp;
    // now trace for remaining workouts backwards
    for (int i = Nu - 2; i >= 0; i--) {
        bestExp = decision[path[i+1]][i+1];
        path[i] = bestExp;
    }

    // check that path is monotonically increasing
    for (int i = 0 ; i < path.size() - 1; i++) {
        assert(path[i] <= path[i+1]);
    }

    return path;
}

bool fit_experience_for_all_users(vector<double>& theta, double **data, int E, vector<vector<int> >& sigma) 
{
    // sigma - set of experience levels for all workouts for all users.. sigma is a matrix.. sigma(u,i) = e_ui i.e experience level of user u at workout i - these values are NOT optimized by L-BFGS.. they are optimized by DP procedure
    int N = data.shape[0];
    int U = get_user_count(data);
    int row = 0;
    double theta_0 = get_theta_0(theta);
    double theta_1 = get_theta_1(theta);
    bool changed = false;
    int index;
    double a_ue, a_e, tprime, diff, t, d;
    for (int u = 0; u < U; u++) {
        int Nu = 0;
        int row_u = row;
        while (row < N && data[row][0] == u) {
            Nu += 1;
            row += 1;
        }
        
        // populate M
        double** M = zeros(E, Nu);
        for (int j = 0; j < Nu; j++) {   // over all workouts for this user
            t = data[row_u + j][3];    // actual time for that workout
            d = data[row_u + j][2];
            for (int i = 0 ; i < E; i++) {      // over all experience levels
                a_ue = get_alpha_ue(theta, u, i, E, index);
                a_e = get_alpha_e(theta, i, E, U, index);
                tprime = (a_e + a_ue) * (theta_0 + theta_1 * d);
                diff = t - tprime;
                M[i][j] = diff * diff;
            }
        }

        double minError;
        bestPath = find_best_path_DP(M, minError);
        //print minError, bestPath
        // update sigma matrix using bestPath
        for (int i = 0 ; i < Nu; i++) {
            if (sigma[u][i] != bestPath[i]) {
                sigma[u][i] = bestPath[i];
                changed = true;
                //print "Updated sigma[%d, %d].." % (u, i)
                //print sigma[u, :]
            }
        }
    }
        
    return changed;
}

map<string, int> get_workouts_per_user(Matrix& data) 
{
    int N = data.shape[0];
    int U = get_user_count(data);
    uins = np.array(range(0, U))
    col0 = data[:, 0]
    u_indices = list(np.searchsorted(col0, uins))
    u_indices.append(N)
    workouts_per_user = [0] * U
    for i in range(0, U):
        start_u = u_indices[i]
        end_u = u_indices[i+1]
        workouts_per_user[i] = end_u - start_u
    return workouts_per_user
}

double F(vector<double>& theta, Matrix& data, double lam1, double lam2, int E, vector<vector<int> >& sigma) 
{
    //from predictor_duration_evolving_user import get_theta_0, get_theta_1, get_alpha_e, get_alpha_ue, get_user_count
    // error function to be minimized
    // assumes data has 4 columns : user_id, user_number, distance, duration and that it is sorted
    // theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    // sigma - set of experience levels for all workouts for all users.. sigma is a matrix.. sigma(u,i) = e_ui i.e experience level of user u at workout i - these values are NOT optimized by L-BFGS.. they are optimized by DP procedure

    //double t1 = time.time();
    int U = get_user_count(data);
    assert(theta.shape[0] == U * E + E + 2);
    int w = 0, i, u, e, index;
    int N = data.shape[0];
    double f = 0;
    double theta_0 = get_theta_0(theta);
    double theta_1 = get_theta_1(theta);
    double a_ue, a_e, d, t, diff;
    while (w < N) {    // over all workouts i.e. all rows in data
        u = (int) (data[w][0]);
        i = 0;   // ith workout of user u
        while (w < N && data[w][0] == u) {
            //e = sigma[u, i]
            e = sigma[u][i];
            a_ue = get_alpha_ue(theta, u, e, E, index);
            a_e = get_alpha_e(theta, e, E, U, index);
            d = data[w, 2];
            t = data[w, 3];
            diff = (a_e + a_ue) * (theta_0 + theta_1*d) - t;
            f += diff * diff;
            w += 1;
            i += 1;
        }
    }

    // divide by denominator
    f = f / (double)N;

    // add regularization 1
    double reg = 0, a_i, a_i_plus_1;
    for (int i = 0; i < E - 1; i++) {
        a_i = get_alpha_e(theta, i, E, U);
        a_i_plus_1 = get_alpha_e(theta, i + 1, E, U, index);
        diff = a_i - a_i_plus_1;
        reg += diff * diff;
        for (int u = 0; u < U; u++) {
            diff = get_alpha_ue(theta, u, i, E, index) - get_alpha_ue(theta, u, i+1, E, index);
            reg += diff * diff;
        }
    }
    f += lam1 * reg;

    // add regularization 2
    double reg2 = 0, a_ui = 0.0;
    for (int i = 0 ; i < E; i++) {
        a_i = get_alpha_e(theta, i, E, U, index);
        reg2 += a_i * a_i;
        for (int u = 0; u < U; u++) {
            a_ui = get_alpha_ue(theta, u, i, E, index);
            reg2 += a_ui * a_ui;
        }
    }
    reg2 += theta_1 * theta_1;
    f += lam2 * reg2;
    
    //double t2 = time.time()
    //print "F = %f, time taken = %f" % (f, t2 - t1)
    return f
}

void Fprime(vector<double>& theta, Matrix& data, double lam1, double lam2, int E, vector<vector<int> >& sigma, double* dE) 
{
    //from predictor_duration_evolving_user import get_theta_0, get_theta_1, get_alpha_e, get_alpha_ue, get_user_count
    // theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    // sigma - set of experience levels for all workouts for all users.. sigma is a matrix.. sigma(u,i) = e_ui i.e experience level of user u at workout i - these values are NOT optimized by L-BFGS.. they are optimized by DP procedure
     //double t1 = time.time()
    int N = data.shape[0];
    int U = get_user_count(data);
    assert(theta.shape[0] == U * E + E + 2);
    double theta_0 = get_theta_0(theta);
    double theta_1 = get_theta_1(theta);

    //np.ndarray[DTYPE_t, ndim=1] dE = np.array([0.0] * theta.shape[0]);
    clear(dE, U * E + E + 2);

    int w = 0, u, i, k, a_uk_index, a_k_index;
    double a_uk, a_k, d, t, t_prime, delta, t0_t1_d, dEda, dE0;
    while (w < N) {    //
        u = data[w][0];
        i = 0;
        while (w < N && data[w][0] == u) {        // over all workouts for the current user
            k = sigma[u][i]; 
            a_uk = get_alpha_ue(theta, u, k, E, a_uk_index);
            a_k = get_alpha_e(theta, k, E, U, a_k_index);

            d = data[w][2];
            t = data[w][3];
            
            t0_t1_d = (theta_0 + theta_1*d);
            t_prime = (a_k + a_uk) * t0_t1_d;
            dEda = 2 * (t_prime - t) * t0_t1_d;

            // dE / d_alpha_k
            dE[a_k_index] += dEda;
            
            // dE / d_alpha_uk
            dE[a_uk_index] += dEda;

            // dE / d_theta_0 and 1
            dE0 = 2 * (t_prime - t) * (a_k + a_uk);
            dE[-2] += dE0;
            dE[-1] += dE0 * d;
            
            w += 1;
            i += 1;
        }
    }

    // divide by denominator
    dE = dE / N;

    // regularization 1 and 2
    double a_k_1, a_uk_1;
    for (int k = 0 ; k < E; k++) {
        a_k = get_alpha_e(theta, k, E, U, a_k_index);
        if (k < E - 1) {
            a_k_1 = get_alpha_e(theta, k + 1, E, U, index);
            dE[a_k_index] +=  2 * lam1 * (a_k - a_k_1);
        }
        if (k > 0) {
            a_k_1 = get_alpha_e(theta, k - 1, E, U, index);
            dE[a_k_index] -=  2 * lam1 * (a_k_1 - a_k);
        }
        dE[a_k_index] += 2 * lam2 * a_k;     // regularization 2

        for (int u = 0; u < U; u++) {
            a_uk = get_alpha_ue(theta, u, k, E, a_uk_index);
            if (k < E - 1) {
                a_uk_1 = get_alpha_ue(theta, u, k+1, E, index);
                dE[a_uk_index] +=  2 * lam1 * (a_uk - a_uk_1);
            }
            if (k > 0) {
                a_uk_1 = get_alpha_ue(theta, u, k-1, E, index);
                dE[a_uk_index] -= 2 * lam1 * (a_uk_1 - a_uk);
            }
            dE[a_uk_index] += 2 * lam2 * a_uk;   // regularization 2
        }
    }
    dE[-1] += 2 * lam2 * theta_1 ;       // regularization 2

    //double t2 = time.time()
    //print "F prime : time taken = ", t2 - t1
} 

void learn(char *infile, double lam1, double lam2, char* outfile)
{
    int N, E = 3;
    Matrix data = read_matrix(infile, N);

    printf("@E = %d,lam1 = %f,lam2 = %f", E, lam1, lam2);
    int U = get_user_count(data);
    //double *theta = new double[U * E + E + 2];
    vector<double> theta(U * E + E + 2);
    init_random(theta);
    
    map<string, int> workouts_per_user = get_workouts_per_user(data);
    vector<vector <int> > sigma(U);
    for (int u = 0; u < U; u++) {
        //sigma.append(list(np.sort(randomState.randint(low = 0, high = E, size = (workouts_per_user[u])))))
        int Nu = workouts_per_user[u];
        vector<int> v(Nu);
        for (int j = 0 ; j < Nu; j++) {
            v[j] = randint(0, E);
        }
        sort(v.begin(), v.end());
        sigma[u] = v;
    }

    /*check grad first
    if (check_grad == True):
        print "Checking gradient.."
        our_grad = np.linalg.norm(Fprime_fn(theta, data, lam1, lam2, E, sigma), ord = 2)
        numerical = np.linalg.norm(scipy.optimize.approx_fprime(theta, F_fn, np.sqrt(np.finfo(np.float).eps), data, lam1, lam2, E, sigma), ord = 2)
        ratio = our_grad / numerical
        print "Ratio = ", ratio
        assert(abs(1.0 - ratio) < 1e-4)
        sys.exit(0)
    */

    int n_iter = 0;
    bool changed = true;
    while (changed && n_iter < 100) {
        printf("Iteration %d..", n_iter);

        // 1. optimize theta
        //[theta, E_min, info] = scipy.optimize.fmin_l_bfgs_b(F_fn, theta, Fprime_fn, args = (data, lam1, lam2, E, sigma),  maxfun=100, maxiter=100, iprint=1, disp=0)
        assert(false);

        // 2. use DP to fit experience levels
        changed = fit_experience_for_all_users(theta, data, E, sigma);

        printf("@E = %lf", E_min);
        n_iter += 1;
    }

    //print "norm of final theta = ", np.linalg.norm(theta, ord = 2)
    //print "final value of error function = ", F_pyx(theta, data, lam1, lam2, E, sigma)
    //print "final value of norm of gradient function = ", np.linalg.norm(Fprime_pyx(theta, data, lam1, lam2, E, sigma), ord = 2)

    return theta, sigma, E
}

int main(int argc, char* argv[])
{
    srand(12345);
    // arguments will have data file name, lam1, lam2, out file name
    assert(argc == 5);
    char* infile = argv[1];
    double lam1 = atof(argv[2]);
    double lam2 = atof(argv[3]);
    char* outfile = argv[4];
    learn(infile, lam1, lam2, outfile);

}
