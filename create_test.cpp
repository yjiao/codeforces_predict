/* 
 * CREATE INPUTS AND OUTPUTS SUITABLE FOR TRAINING CF RNN
 *
 * INPUTS:
 *    problem_ratings.csv
 *    all_ratingHistories.csv
 *    all_submissions.csv
 *
 *  OUTPUTS:
 *    X
 *        DIMENSIONS:
*            REAL NUMBERS
 *            - User rating at time point (normalized by MAX_RATING)
 *            - Problem rating (normalized by MAX_RATING)
 *            - Problem points awarded (normalized by MAX_RATING)
 *            - Time elapsed from contest (normalized by MAX_SECONDS, in
 *              reverse time order, st the most recent contest has the smallest
 *              number)
*            INDICATOR VARIABLES
 *            - Verdict
 *            - Problem Tags
 *            - Participant type
 *    Y
 *      - deltas rating corresponding to each contest
 *
 *
 *  For each user, "cut" timepoints st the beginning of a vector contains
 *  the most recent actions of a user immediately before a contest
 *
 *  MIN_ENTRIES: the minimum number of entries in a dataset for us to consider keeping it
 *  MAX_ENTRIES: the maximum number of entries allowed in a dataset
 *  MAX_SECONDS: the maximum amount of time elapsed between a submission and the corresponding contest
*                Note: this allows us to normalize the time variable in a meaningful way
 *  MAX_RATING: normalization constant for all rating-related values
 *  MAX_PTS: normalization constant for all point-related values
 *
 *
 *  IMPLEMENTATION NOTES
 *    - file pointer moves monotonically forwards
 */
#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <set>
#include <queue>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <memory.h>
#include <string>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cassert>

using namespace std;

typedef long long LL;
typedef pair<int,int> PII;

#define MP make_pair
#define PB push_back
#define FF first
#define SS second

#define FORN(i, n) for (int i = 0; i <  (int)(n); i++)
#define FOR1(i, n) for (int i = 1; i <= (int)(n); i++)
#define FORD(i, n) for (int i = (int)(n) - 1; i >= 0; i--)

#define DEBUG(X) { cout << #X << " = " << (X) << endl; }
#define PR0(A,n) { cout << #A << " = "; FORN(_,n) cout << A[_] << ' '; cout << endl; }

// #define FL fflush(stdout)

#define MOD 1000000007
#define INF 2000000000

int GLL(LL& x) {
  return scanf("%lld", &x);
}

int GI(int& x) {
  return scanf("%d", &x);
}

const int MIN_ENTRIES = 50;
const int MAX_ENTRIES = 1000;
const int MAX_SECONDS = 30 * 24 * 3600;
const int MAX_RATING = 3500;
const int MAX_PTS = 5000;

int vector_ptr = 0;
queue<int> q;

// split a string given a char delimiter, IN PLACE
void split(const string &input, char delim, vector<string> &output) {
    // split a string "input" based on delimiter, return vector "output"
    stringstream input_stream;
    input_stream.str(input);
    string token;
    while (getline(input_stream, token, delim)) {
      output.PB(token);
    }
}

void read_line(ifstream &fh, vector<string> &tokens, char delim) {
  // header
  string line;
  getline(fh, line);
  split(line, delim, tokens);
}

void add_tokens_to_df(vector<string> &tokens, vector<string> &header, map<string, vector<string> > &df) {
  FORN(i, tokens.size()) {
    df[ header[i] ].PB(tokens[i]);
  }
}

// move file pointer to right AFTER the start of a new contest 
void find_next_contest(ifstream &fh, vector<string> &header, map<string, vector<string> > &df) {

  int idx = df["participantType"].size();

  string cur_contest;
  string next_contest;
  string cur_type;

  // move file ptr forward until we find another contest
  do {
    vector<string> line;
    read_line(fh, line, '\t');
    add_tokens_to_df(line, header, df);

    cur_contest = df["contestID"].back();
    cur_type = df["participantType"].back();

  } while (cur_type != "CONTESTANT");

  // move file ptr forward until we're out of the current contest
  do {
    vector<string> line;
    read_line(fh, line, '\t');
    add_tokens_to_df(line, header, df);

    next_contest = df["contestID"].back();
    cur_type = df["participantType"].back();

  } while (cur_contest == next_contest);

  printf("--------------------------------\n");
  printf("Prev contest: %s\n", cur_contest.c_str());
  printf("Next contest: %s\n", next_contest.c_str());

  // TODO: check for all conditions
  string prev_found_contestID = "none";
  FORN(i, MAX_ENTRIES) {
    vector<string> line;
    read_line(fh, line, '\t');
    add_tokens_to_df(line, header, df);  // TODO: make this actually wrap around
    if ( df["participantType"][vector_ptr] == "CONTESTANT" && prev_found_contestID != df["contestID"][vector_ptr]) {
      q.push( vector_ptr );
      prev_found_contestID = df["contestID"][vector_ptr];
    }
    printf("contest: %s %s\n", df["contestID"].back().c_str(), df["participantType"].back().c_str());
  }
//  for (auto kv : df) { printf("%d, %s: %s\n", kv.SS.size(), kv.FF.c_str(), kv.SS.back().c_str()); }
}

int main() {
  string prefix = "/Users/Joy/Dropbox/Algorithms/codeforces/codeforces-api/prediction/";
  string fn_submission = "all_submissions.tsv";
  ifstream fh(prefix+fn_submission);

  vector<string> submission_hdr;
  read_line(fh, submission_hdr, '\t');

  map<string, vector<string> > df_sub;
  vector<string> line;
  read_line(fh, line, '\t');
  add_tokens_to_df(line, submission_hdr, df_sub);

  find_next_contest(fh, submission_hdr, df_sub);

  printf("\n");
  FORN(i, q.size()) {
    printf("%d\n", q.front());
    q.pop();
  }
  printf("\n");

//  FORN(j, 10) {
//    printf("---------------------------------------\n");
//    for (auto kv : df_sub) {
//      printf("%s: %s\n", kv.FF.c_str(), kv.SS[j].c_str());
//    }
//  }
  cout << endl;
  fh.close();

  return 0;
}
