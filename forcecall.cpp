// 2016 11 09
// Yunxin Joy Jiao
// Given samtools pileups, count the number of reference vs alt reads
// Note: in later scripts, we need to put in the following missing headers:
//            * build (37)
//            * tumor_barcode
//            * normal_barcode
//            * streka/vardict/mutect call stats
// These functions are implemented in postProcess.R
//
// 2016 11 17
// Added simple loop to make VCF format conform to maflite format needed for oncotator
// Ex. VCF ref AGC, alt A
//	   MAF ref  GC, alt -
// Also update start idx when this adjustment needs to be made

#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <set>
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
#include <queue>
#include <string>
#include <unistd.h>
#include <cassert>

using namespace std;

typedef long long LL; 
typedef pair<int,int> PII;
typedef pair<PII,char> PIIC;

#define MP make_pair
#define PB push_back
#define FF first
#define SS second

#define FORN(i, n) for (int i = 0; i <  (int)(n); i++)
#define ALL(c) (c).begin(), (c).end()
#define FOR1(i, n) for (int i = 1; i <= (int)(n); i++)
#define FORD(i, n) for (int i = (int)(n) - 1; i >= 0; i--)
#define FOREACH(i, c) for (typeof((c).begin()) i = (c).begin(); i != (c).end(); i++)

// struct to keep track of which callers found a mutation
typedef struct {
    uint8_t count;
    bool mutect;
    bool strelka;
    bool vardict;
} callstats_t;

// struct used to store information relating to mutations
typedef struct {
    uint8_t chr;  // for now, ignore X, Y, contigs
    uint64_t start;
    uint64_t end;
    string ref;
    string alt;
} mutation_t;

// custom compare function used for keeping mutation_t objects in ordered sets
struct mutation_compare {
    // custom compare function for mutation_t objects
    bool operator() (const mutation_t &lhs, const mutation_t &rhs) const {
	// returns true if lhs is before rhs
	if (lhs.chr < rhs.chr) return true;
	if (lhs.chr > rhs.chr) return false;

	// now they have the same chr
	if (lhs.start < rhs.start) return true;
	if (lhs.start > rhs.start) return false;

	// now they have the same chr and same start
	if (lhs.end < rhs.end) return true;
	if (lhs.end > rhs.end) return false;

	// same chr, start, end, so the ref seq will also be the same
	if (lhs.alt < rhs.alt) return true;
	return false;
    }
};

// split a string given a char delimiter, returns a vector of strings
vector<string> split(const string &input, char delim) {
    vector<string> output;
    // split a string "input" based on delimiter, return vector "output"
    stringstream input_stream;
    input_stream.str(input);
    string token;
    while (getline(input_stream, token, delim)) {
	output.PB(token);
    }
    return output;
}

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

// constructor for mutation_t objects
mutation_t new_mutation(uint8_t chr, uint64_t start, uint64_t end, string ref, string alt) {
    mutation_t mut;
    mut.chr = chr;
    mut.start = start;
    mut.end = end;
    mut.ref = ref;
    mut.alt = alt;
    return mut;
}

// write a single line for a force-called mutation to ofstream
static inline void write_line(ofstream &fh, mutation_t &mut, 
	pair<uint32_t, uint32_t> &counts) {

    // first, we need to fix the difference between VCF designation of indels
    // vs maflite
    // Ex. VCF ref AGC, alt A
    //	   MAF ref  GC, alt -

    if (mut.ref.size() != mut.alt.size()) {
	size_t idx = 0;
	while (mut.ref[idx] == mut.alt[idx]) {
	    idx++;
	}
	mut.ref = mut.ref.substr(idx);
	mut.alt = mut.alt.substr(idx);

	if (mut.alt.size() == 0) mut.alt = "-";
	if (mut.ref.size() == 0) mut.ref = "-";
	mut.start += idx;
    }

    double freq = double(counts.SS) / (double(counts.SS) + double(counts.FF));
    fh << to_string(mut.chr) << '\t';    // chr
    fh << mut.start << '\t';             // start
    fh << mut.end << '\t';               // end
    fh << mut.ref << '\t';               // ref_allele
    fh << mut.ref << '\t';               // tum_allele1
    fh << mut.alt << '\t';               // tum_allele2
    fh << freq << '\t';                  // tumor_f
    fh << counts.SS << '\t';             // t_alt_count
    fh << counts.FF;                     // t_ref_count
    fh << "\n";                          // 
}

// write a single header line (fields correspond to write_line function)
static inline void write_header(ofstream &fh) {
    fh << "chr" << '\t';
    fh << "start" << '\t';
    fh << "end" << '\t';
    fh << "ref_allele" << '\t';
    fh << "tum_allele1" << '\t';
    fh << "tum_allele2" << '\t';
    fh << "tumor_f" << '\t';
    fh << "t_alt_count" << '\t';
    fh << "t_ref_count";
    fh << "\n";
}

// opens the .bamlist file, populates a vector of strings of paths to bam files, IN PLACE
static void parse_bamlist(string path, vector<string> &samples) {
    ifstream fh(path);
    string line, filename;
    while (!fh.eof()) {
	getline(fh, line);
	vector<string> tokens;
	split(line, '/', tokens);
	if (tokens.size()) {
	    filename = tokens.back();
	    filename.replace(filename.end()-4, filename.end(), "");
	    samples.PB(filename);
	}
    }
    fh.close();
}

// count the number of appearances of (string) pattern in given string
uint32_t count_substr(const string &str, const string pattern) {
    size_t i = 0;
    uint32_t cnt = 0;
    while (str.find(pattern, i) != string::npos) {
	cnt++;
	i = str.find(pattern, i) + pattern.size();
    }
    return cnt;
}

// get alt count for a mutation and pileup
void count_alt(mutation_t &mut, vector<string> &pileup, uint8_t nSamps,
	vector< pair<uint32_t, uint32_t> > &counts) {
    // indel
    // count alt reads only, since this occurs before the actual mutation occurs
    assert(mut.alt.size() != mut.ref.size());
    uint8_t chr = stoi(pileup[0]);
    uint64_t idx = stoi(pileup[1]);
    assert(mut.chr == chr); assert(mut.start <= idx); assert(mut.end >= idx);

    FORN(j, nSamps) {
	int i = j*3 + 3;
	uint32_t depth = stoi(pileup[i]);
	string reads = pileup[i+1];
	string quals = pileup[i+2];

	string altstr;
	if (mut.alt.size() > mut.ref.size()) {
	//	      printf("INSERTION: %s -> %s\n", mut.ref.c_str(), mut.alt.c_str());
	  altstr = mut.alt.substr( mut.ref.size(), mut.alt.size() );
	  altstr = "+" + to_string(altstr.size()) + altstr;
	} else {
	//	      printf("DELETION: %d %d %s -> %s\n", mut.chr, mut.start, mut.ref.c_str(), mut.alt.c_str());
	//	      printf("      at: %d %d\n", chr, idx);
	  altstr = mut.ref.substr( mut.alt.size(), mut.ref.size() );
	  altstr = "-" + to_string(altstr.size()) + altstr; 
	}

	transform(altstr.begin(), altstr.end(), altstr.begin(), ::toupper);
	counts[j].SS += count_substr(reads, altstr);
	transform(altstr.begin(), altstr.end(), altstr.begin(), ::tolower);
	counts[j].SS += count_substr(reads, altstr);
    }
}

// get ref count for a mutation and pileup
void count_ref(mutation_t &mut, vector<string> &pileup, uint8_t nSamps,
	vector< pair<uint32_t, uint32_t> > &counts) {
    // indel
    // count refs only
    assert(mut.alt.size() != mut.ref.size());
    uint8_t chr = stoi(pileup[0]);
    uint64_t idx = stoi(pileup[1]);
    assert(mut.chr == chr); assert(mut.start <= idx); assert(mut.end >= idx);

    FORN(j, nSamps) {
	int i = j*3 + 3;
	uint32_t depth = stoi(pileup[i]);
	string reads = pileup[i+1];
	string quals = pileup[i+2];

	// count ref reads
	counts[j].FF += count(reads.begin(), reads.end(), '.');
	counts[j].FF += count(reads.begin(), reads.end(), ',');
    }
}

// simple case of getting alt/ref counts for a SNV
void count_ref_alt_snv(mutation_t &mut, vector<string> &pileup, uint8_t nSamps,
	 vector< pair<uint32_t, uint32_t> > &counts) {
    // simple SNV
    assert(mut.alt.size() == mut.ref.size());
    uint8_t chr = stoi(pileup[0]);
    uint64_t idx = stoi(pileup[1]);
    assert(mut.chr == chr); assert(mut.start <= idx); assert(mut.end >= idx);

    FORN(j, nSamps) {
	int i = j*3 + 3;
	uint32_t depth = stoi(pileup[i]);
	if (depth) {
	    string reads = pileup[i+1];
	    string quals = pileup[i+2];

	    // count ref reads
	    counts[j].FF += count(reads.begin(), reads.end(), '.');
	    counts[j].FF += count(reads.begin(), reads.end(), ',');

	    assert(mut.alt.size() == 1 & mut.ref.size() == 1);
	    counts[j].SS += count(reads.begin(), reads.end(), toupper(mut.alt[0]));
	    counts[j].SS += count(reads.begin(), reads.end(), tolower(mut.alt[0]));
	
	}
    }
}

// divide each element of a vector pair by int b
static void vec_pair_divide( vector< pair<uint32_t, uint32_t> > &A, int b) {
    FORN(i, A.size()) {
	A[i].FF /= b;
    }
}

// add elements of vector B to vector A, IN PLACE, alters A
static void vec_pair_add( vector< pair<uint32_t, uint32_t> > &A,  vector< pair<uint32_t, uint32_t> > &B) {
    // add B to A
    assert(A.size() == B.size());
    FORN(i, A.size()) {
	A[i].FF += B[i].FF;
	A[i].SS += B[i].SS;
    }
}

// prints fields in mutation_t, for debugging
static inline void print_mut(mutation_t &mut) {
    printf("chr %d %d %d %s %s\n",
	    mut.chr,
	    mut.start,
	    mut.end,
	    mut.ref.c_str(),
	    mut.alt.c_str());
}

// get a single line in the pileup corresponding to the current position, assumes line exists and pileup is in sorted order
static void get_pileup(mutation_t &mut, ifstream &fh_pileup, size_t nfiles,
    vector< pair<uint32_t, uint32_t> > &counts) {

//    printf("---------------------------------\n");
    uint8_t chr = 0;
    uint64_t idx = 0;
    uint32_t ref = 0;
    uint32_t alt = 0;
    uint32_t depth = 0;
    vector<string> pileup;

    string line;
    while (chr <= mut.chr && idx < mut.start) {
	getline(fh_pileup, line);
	pileup = split(line, '\t');
	chr = stoi(pileup[0]);
	idx = stoi(pileup[1]);
    }
    assert(mut.chr==chr);
    assert(mut.start==idx);
    assert(mut.end>=idx);

    
    // at this point, the current line in the pileup file is at the start of the mutation
    // check whether mutation is SNV or INDEL
    if (mut.alt.size() == mut.ref.size()) {
	count_ref_alt_snv(mut, pileup, nfiles, counts);
    } else {
	uint32_t mutlen = 1;
	count_alt(mut, pileup, nfiles, counts);
	count_ref(mut, pileup, nfiles, counts);
	while (chr == mut.chr && idx < mut.end) {
	    getline(fh_pileup, line);
	    pileup = split(line, '\t');
	    chr = stoi(pileup[0]);
	    idx = stoi(pileup[1]);
	    count_ref(mut, pileup, nfiles, counts);
	    mutlen++;
	}
	//printf("mutlen: %d\n", mutlen);
	vec_pair_divide(counts, mutlen);
    }
    
    //print_mut(mut);
    //for (auto c: counts) {
    //    printf("(%d, %d) ", c.FF, c.SS);
    //}
    //printf("\n");

    // at this point, the current line in the pileup file is at the end of the mutation
    assert(mut.chr==chr);
    assert(mut.end==idx);

    // we are now done counting reads, so we can output one line to each file
}

static void forcecall(string path_mutation, string path_pileup, vector<string> &samples, string prefix_output) {
    ifstream fh_mut(path_mutation);
    ifstream fh_pileup(path_pileup);
    size_t nsamps = samples.size();

    // open a vector of (pointers to) file handles, one for each sample we are analyzing
    for (auto sampid : samples) {
	string path = prefix_output + sampid + ".maf";
	ofstream fh_samp = ofstream(path);
	write_header(fh_samp);
    }
    printf("done writing headers\n");

    string line, header;
    getline(fh_mut, header);  // get rid of header line

    int i = 0;
    while (!fh_mut.eof()) {
	getline(fh_mut, line);
//	printf("read line %s\n", line.c_str());
	vector<string> mtokens;
	split(line, '\t', mtokens);

	if (mtokens.size()) {
//	    printf("make mut... ");
	    mutation_t mut = new_mutation(stoi(mtokens[0]),
					       stoi(mtokens[1]),
			    		       stoi(mtokens[2]),
			    		       mtokens[3],
			    		       mtokens[4]);

//	    printf(" done\n");
	    // initialize a vector to hold pairs of ref/alt counts for each sample
//	    printf("init counts... ");
	    vector< pair<uint32_t, uint32_t> > counts (nsamps, MP(0, 0));
//	    printf(" done\n");

//	    printf("get pileup...");
	    get_pileup(mut, fh_pileup, nsamps, counts);
//	    printf(" done\n");

	    // TODO: repeatedly opening and closing the files is pretty inefficinet
	    // However, there seems to be issues with the file streams going
	    // out of scope in a for loop that malloc can't seem to fix.
	    FORN(i, nsamps) {
		auto sampid = samples[i];
		string path = prefix_output + sampid + ".maf";
		ofstream fh = ofstream(path, ios::app);
		write_line(fh, mut, counts[i]);
		fh.close();
	    }
	    i++;
	}
    }

    printf("%d mutations processed.\n", i);
    fh_mut.close();
    fh_pileup.close();
}

int main(int argc, char* argv[]) {
    string pid = argv[1];
    string prefix_data = argv[2];
    string prefix_mutation = argv[3];
    string prefix_output  = argv[4];

    // add a dash at the end just in case
    prefix_data += "/";
    prefix_output += "/";
    prefix_mutation += "/";

    // create paths for files we need: bamlist and pileup
    string path_bamlist = prefix_data + pid + ".bamlist";
    string path_pileup = prefix_data + pid + ".pileup";
    string path_muts = prefix_mutation + pid + "_pass.maf";

    printf("bamlist: %s\n", path_bamlist.c_str());
    printf("pileup: %s\n", path_pileup.c_str());
    printf("maf: %s\n", path_muts.c_str());

    // get sample names from bamlist
    vector<string> samples;
    parse_bamlist(path_bamlist, samples);
    printf("done parsing bamlist\n");
    forcecall(path_muts, path_pileup, samples, prefix_output);

}