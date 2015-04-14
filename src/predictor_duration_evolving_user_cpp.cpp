#include<string>
#include<fstream>
#include<iostream>
#include<stdlib.h>
#include<sstream>
#include<algorithm>
#include<random>
#include<chrono>
#include<vector>
#include<cassert>
#include<lbfgs.h>
#include <dlib/matrix.h>
#include <dlib/optimization.h>

using namespace std;
typedef dlib::matrix<double,0,1> column_vector;

double** alloc_matrix(int N, int M);
void free_matrix(double **mat, int nrows, int ncols);

class Matrix
{
    public:
        int shape[2];
        double **data;

        Matrix(int nrows, int ncols, bool allocate = true)
        {
            if (allocate) 
                data = alloc_matrix(nrows, ncols);
            shape[0] = nrows;
            shape[1] = ncols;
        }

        double* operator[](int i)
        {
            return data[i];
        }

        void free()
        {
            free_matrix(data, shape[0], shape[1]);
        }
};

double** alloc_matrix(int N, int M)
{
    double **mat = new double*[N];
    for (int i = 0; i < N; i++)
        mat[i] = new double[M];
    return mat;
}

void free_matrix(double **mat, int nrows, int ncols)
{
    for (int i = 0 ; i < nrows; i++) {
        delete[] mat[i];
    }
    delete[] mat;
}

double** zeros(int nrows, int ncols)
{
    double **z = alloc_matrix(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0 ; j < ncols; j++) {
            z[i][j] = 0;
        }
    }
    return z;
}

Matrix zeros_matrix(int nrows, int ncols)
{
    Matrix m(nrows, ncols);
    for (int i = 0 ; i < nrows; i++) {
        for (int j = 0 ; j < ncols; j++) {
            m.data[i][j] = 0;
        }
    }
    return m;
}

Matrix read_matrix(const char *infile, int& N)
{
    // count lines
    N = 0;
    ifstream f;
    f.open(infile);
    string line;
    string token;
    int ncols = 0;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        if (ncols == 0) {
            while (iss >> token)
                ncols++;
        }
        N++;
    }
    cout << "Data matrix dimensions = " << N << " x " << ncols << "\n";
    f.close();

    f.open(infile);
    Matrix data(N, ncols);
    int i = 0;
    while (std::getline(f, line)) {
        std::istringstream iss(line);
        for (int j = 0 ; j < ncols; j++) {
            iss >> token;
            data[i][j] = stod(token);
        }
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

void init_random(double* a, int n)
{
    for (int i = 0; i < n; i++) {
        a[i] = ((double)rand()/(double)RAND_MAX);
    }
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

double get_alpha_ue(const double* theta, int u, int e, int E, int& index)
{
    // theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    index = u * E + e;
    return theta[index];
}

double get_alpha_e(const double* theta, int e, int E, int U, int& index) 
{
    // theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    index = U * E + e;
    return theta[index]; 
}

double get_theta_0(const double* theta, int nparams)
{
    return theta[nparams - 2];
}

double get_theta_1(const double* theta, int nparams)
{
    return theta[nparams - 1];
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
            /* this code segment considers all experience levels permitted, not just the curent and previous one. Did not improve results.
            double o_best = numeric_limits<double>::max();
            for (int i = 0; i <= m; i++) {
                double o = M[m][n] + D[i][n - 1];
                if (o < o_best) {
                    o_best = o;
                    D[m][n] = o_best;
                    decision[m][n] = i;
                }
            }*/
            
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
    for (unsigned int i = 0 ; i < path.size() - 1; i++) {
        assert(path[i] <= path[i+1]);
    }
    free_matrix(D, E, Nu);
    free_matrix(decision, E, Nu);
    return path;
}

bool fit_experience_for_all_users(const double *theta, int nparams, Matrix& data, int E, vector<vector<int> >& sigma) 
{
    // sigma - set of experience levels for all workouts for all users.. sigma is a matrix.. sigma(u,i) = e_ui i.e experience level of user u at workout i - these values are NOT optimized by L-BFGS.. they are optimized by DP procedure
    int N = data.shape[0];
    int U = get_user_count(data);
    int row = 0;
    double theta_0 = get_theta_0(theta, nparams);
    double theta_1 = get_theta_1(theta, nparams);
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
        Matrix M = zeros_matrix(E, Nu);
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
        vector<int> bestPath = find_best_path_DP(M, minError);
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

        M.free();
    }
        
    return changed;
}

vector<int> get_workouts_per_user(Matrix& data) 
{
    int N = data.shape[0];
    int U = get_user_count(data);
    vector<int> workouts_per_user(U, 0);
    for (int i = 0 ; i < N; i++) {
        int u = data[i][0];
        workouts_per_user[u]++;
    }
    return workouts_per_user;
    /*
    uins = np.array(range(0, U))
    col0 = data[:, 0]
    u_indices = list(np.searchsorted(col0, uins))
    u_indices.append(N)
    workouts_per_user = [0] * U
    for i in range(0, U):
        start_u = u_indices[i]
        end_u = u_indices[i+1]
        workouts_per_user[i] = end_u - start_u
    return workouts_per_user;
    */
}

double F(const double* theta, Matrix& data, double lam1, double lam2, int E, vector<vector<int> >& sigma) 
{
    //from predictor_duration_evolving_user import get_theta_0, get_theta_1, get_alpha_e, get_alpha_ue, get_user_count
    // error function to be minimized
    // assumes data has 4 columns : user_id, user_number, distance, duration and that it is sorted
    // theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    // sigma - set of experience levels for all workouts for all users.. sigma is a matrix.. sigma(u,i) = e_ui i.e experience level of user u at workout i - these values are NOT optimized by L-BFGS.. they are optimized by DP procedure

    //double t1 = time.time();
    int U = get_user_count(data);
    int nparams = (U * E + E + 2);
    int w = 0, i, u, e, index;
    int N = data.shape[0];
    double f = 0;
    double theta_0 = get_theta_0(theta, nparams);
    double theta_1 = get_theta_1(theta, nparams);
    double a_ue, a_e, d, t, diff;
    while (w < N) {    // over all workouts i.e. all rows in data
        u = (int) (data[w][0]);
        i = 0;   // ith workout of user u
        while (w < N && data[w][0] == u) {
            //e = sigma[u, i]
            e = sigma[u][i];
            a_ue = get_alpha_ue(theta, u, e, E, index);
            a_e = get_alpha_e(theta, e, E, U, index);
            d = data[w][2];
            t = data[w][3];
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
        a_i = get_alpha_e(theta, i, E, U, index);
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
    return f;
}

void Fprime(const double* theta, Matrix& data, double lam1, double lam2, int E, vector<vector<int> >& sigma, double* dE) 
{
    //from predictor_duration_evolving_user import get_theta_0, get_theta_1, get_alpha_e, get_alpha_ue, get_user_count
    // theta - first UxE elements are per-user per-experience alpha values, next E elements are per experience offset alphas, last 2 are theta0 and theta1
    // sigma - set of experience levels for all workouts for all users.. sigma is a matrix.. sigma(u,i) = e_ui i.e experience level of user u at workout i - these values are NOT optimized by L-BFGS.. they are optimized by DP procedure
     //double t1 = time.time()
    int N = data.shape[0];
    int U = get_user_count(data);
    int nparams = U * E + E + 2;
    double theta_0 = get_theta_0(theta, nparams);
    double theta_1 = get_theta_1(theta, nparams);

    //np.ndarray[DTYPE_t, ndim=1] dE = np.array([0.0] * theta.shape[0]);
    clear(dE, U * E + E + 2);

    int w = 0, u, i, k, a_uk_index, a_k_index;
    double a_uk, a_k, d, t, t_prime, t0_t1_d, dEda, dE0;
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
            dE[nparams - 2] += dE0;
            dE[nparams - 1] += dE0 * d;
            
            w += 1;
            i += 1;
        }
    }

    // divide by denominator
    for (int i = 0; i < nparams; i++) {
        dE[i] = dE[i] / N;
    }

    // regularization 1 and 2
    double a_k_1, a_uk_1;
    int index;
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
    dE[nparams - 1] += 2 * lam2 * theta_1 ;       // regularization 2

    //double t2 = time.time()
    //print "F prime : time taken = ", t2 - t1
}

class Constants
{
    public:
        Matrix* data;
        vector<vector<int> >* sigma;
        int E, nparams;
        double lam1, lam2;
};

static lbfgsfloatval_t evaluate(void *instance, const lbfgsfloatval_t *theta, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step)
{
    lbfgsfloatval_t fx = 0.0;
    Constants *consts = (Constants*) instance;
    //vector<double> theta(x, x + consts -> nparams);
    fx =  F(theta, *(consts -> data), consts -> lam1, consts -> lam2, consts -> E, *(consts -> sigma));
    Fprime(theta, *(consts -> data), consts -> lam1, consts -> lam2, consts -> E, *(consts -> sigma), g);
    return fx;
}

static int progress(void *instance, const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k,int ls )
{
    if (k == 1) {
        printf("Iteration %d: \n", k);
        printf("@fx = %f\n", fx);
    }
    //printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    //printf("\n");
    return 0;
}

void optimize(lbfgsfloatval_t* theta, Matrix& data, double lam1, double lam2, int E, vector<vector<int> >& sigma, lbfgs_parameter_t& param)
{
    // = scipy.optimize.fmin_l_bfgs_b(F_fn, theta, Fprime_fn, args = (data, lam1, lam2, E, sigma),  maxfun=100, maxiter=100, iprint=1, disp=0)
    lbfgsfloatval_t fx = 0;
    int U = get_user_count(data);
    Constants c;
    c.data = &data;
    c.sigma = &sigma;
    c.E = E;
    c.lam1 = lam1;
    c.lam2 = lam2;
    int ret = lbfgs(U * E + E + 2, theta, &fx, evaluate, progress, (void *)&c, &param);
    printf("LBFGS terminated with status %d\n", ret);
}

double F_dlib(Constants c, const column_vector x)
{
    lbfgsfloatval_t *theta = new lbfgsfloatval_t[x.nr()];
    for (int i = 0 ; i < x.nr(); i++)
        theta[i] = x(i, 0);
    double f = F(theta, *(c.data), c.lam1, c.lam2, c.E, *(c.sigma));
    delete[] theta;
    return f;
}

void learn(char *infile, double lam1, double lam2, char* outfile, int E)
{
    int N;
    Matrix data = read_matrix(infile, N);
    bool check_grad = false;

    printf("@E = %d,lam1 = %f,lam2 = %f\n", E, lam1, lam2);
    int U = get_user_count(data);
    int nparams = U * E + E + 2;
    cout << "U = " << U << " , E = " << E << " , nparams = " << nparams << "\n";
    //double *theta = new double[U * E + E + 2];
    lbfgsfloatval_t *theta = lbfgs_malloc(nparams);
    //vector<double> theta(th, th + U * E + E + 2);
    init_random(theta, nparams);
    lbfgs_parameter_t lbfgsparam;
    lbfgs_parameter_init(&lbfgsparam);
    //lbfgsparam.epsilon = 1e-7;
    lbfgsparam.m = 10;
    
    vector<int> workouts_per_user = get_workouts_per_user(data);
    vector<vector <int> > sigma(U);
    for (int u = 0; u < U; u++) {
        //sigma.append(list(np.sort(randomState.randint(low = 0, high = E, size = (workouts_per_user[u])))))
        int Nu = workouts_per_user[u];
        vector<int> v(Nu);
        for (int j = 0 ; j < Nu; j++) {
            v[j] = randint(0, E - 1);
            assert(v[j] >= 0 && v[j] < E);
        }
        sort(v.begin(), v.end());
        sigma[u] = v;
    }

    if (check_grad) {
        printf("Checking gradient before training..\n");
        column_vector m(nparams);
        for (int i = 0 ; i < nparams; i++) {
            m(i, 0) = theta[i];
        }

        // Analytic
        //ourgrad = np.linalg.norm(eprime_fn(np.array(theta), train, lam), ord = 2)
        //print "gradient = ", ourgrad
        lbfgsfloatval_t *dE = new lbfgsfloatval_t[nparams];
        Fprime(theta, data, lam1, lam2, E, sigma, dE);
        double analytic = 0;
        for (int i = 0 ; i < nparams; i++) {
            analytic += dE[i] * dE[i];
        }
        analytic = sqrt(analytic);
        cout << "analytic = " << analytic << "\n";
        delete[] dE;

        // Numerical
        Constants c;
        c.data = &data;
        c.lam1 = lam1;
        c.lam2 = lam2;
        c.sigma = &sigma;
        c.E = E;
        column_vector numerical_grad_vec = dlib::derivative(F_dlib)(c, m);
        double numerical = 0;
        for (int i = 0; i < nparams; i++) {
            double vi = numerical_grad_vec(i, 0);
            numerical += vi * vi;
        }
        numerical = sqrt(numerical);
        cout << "numerical = " << numerical << "\n";

        double ratio = analytic / numerical;
        printf("Ratio = %lf\n", ratio);
        cout << "Ratio = " << ratio << "\n";
        cout << fabs(1.0 - ratio) << "\n";
        assert(fabs(1.0 - ratio) < 1e-6);
        exit(0);
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
        printf("Super Iteration %d..", n_iter);

        // 1. optimize theta
        //[theta, E_min, info] = scipy.optimize.fmin_l_bfgs_b(F_fn, theta, Fprime_fn, args = (data, lam1, lam2, E, sigma),  maxfun=100, maxiter=100, iprint=1, disp=0)
        optimize(theta, data, lam1, lam2, E, sigma, lbfgsparam);

        // 2. use DP to fit experience levels
        changed = fit_experience_for_all_users(theta, nparams, data, E, sigma);

        //printf("@E = %lf", E_min);
        n_iter += 1;
    }

    //print "norm of final theta = ", np.linalg.norm(theta, ord = 2)
    //print "final value of error function = ", F_pyx(theta, data, lam1, lam2, E, sigma)
    //print "final value of norm of gradient function = ", np.linalg.norm(Fprime_pyx(theta, data, lam1, lam2, E, sigma), ord = 2)

    //return theta, sigma, E
    ofstream of;
    of.open(outfile);
    of << "E=" << E << "\n";
    of << "theta=[";
    int i = 0;
    for (i = 0 ; i < nparams - 1; i++)
        of << theta[i] << ",";
    of << theta[i] << "]\n";
    of << "sigma=[";
    int n_users = sigma.size();
    vector<int> sigma_u;
    for (int i = 0 ; i < n_users; i++) {
        of << "[";
        sigma_u = sigma[i];
        int j = 0;
        for (j = 0 ; j < (int)sigma_u.size() - 1; j++)
            of << sigma_u[j] << ",";
        of << sigma_u[j] << "]";
        if (i < n_users - 1)
            of << ",";
    }
    of << "]";
    of.close();
    lbfgs_free(theta);
    data.free();
}

int main(int argc, char* argv[])
{
    srand(12345);
    // arguments will have data file name, lam1, lam2, out file name
    assert(argc == 6);
    char* infile = argv[1];
    double lam1 = atof(argv[2]);
    double lam2 = atof(argv[3]);
    char* outfile = argv[4];
    int E = atoi(argv[5]);
    learn(infile, lam1, lam2, outfile, E);
}
