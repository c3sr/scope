
#include "b5_papi.hpp"

#include <cstring> // watchout, there's a compile error with -pedantic
#include <papi.h>  // @see http://stackoverflow.com/questions/36164163/pedantic-raising-error-when-linking-papi

#include <cmath>
#include <cstdio>
#include <cstdlib>

//-----------------------------------------------------------------------------

#ifndef B5_LOG
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments" // this works in all modern compilers I've tested
#   define B5_LOG(msg, ...) printf(msg, ## __VA_ARGS__)
#   pragma clang diagnostic pop
#endif

#ifndef B5_ERROR
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments" // this works in all modern compilers I've tested
#   define B5_ERROR_FL(file, line, msg, ...) { fprintf(stderr, "%s:%d: ERROR: " msg, file, line, ## __VA_ARGS__); abort(); }
#   define B5_ERROR(msg, ...) B5_ERROR_FL(__FILE__, __LINE__, msg, ## __VA_ARGS__)
#   pragma clang diagnostic pop
#endif

#ifndef B5_WARNING
#   pragma clang diagnostic push
#   pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments" // this works in all modern compilers I've tested
#   define B5_WARNING_FL(file, line, msg, ...) { fprintf(stderr, "%s:%d: WARNING: " msg, file, line, ## __VA_ARGS__);/* abort(); */}
#   define B5_WARNING(msg, ...) B5_WARNING_FL(__FILE__, __LINE__, msg, ## __VA_ARGS__)
#   pragma clang diagnostic pop
#endif

#ifndef B5_ASSERT
#   ifndef _NDEBUG
#       define B5_ASSERT(cond) { if(!(cond)) { B5_ERROR("assertion %s failed.\n", #cond); } }
#   else
#       define B5_ASSERT(cond)
#   endif
#endif

//-----------------------------------------------------------------------------

#ifndef B5_PAPI_ERR
#   define B5_PAPI_ERR(err) \
{\
    if((err) != PAPI_OK) \
    {\
        B5_ERROR("PAPI error: %s\n", PAPICounters::_report_err((err)));\
    }\
}
#endif

#ifndef B5_CHECK_PAPI_CALL
#   define B5_CHECK_PAPI_CALL(call) \
{\
    if(int _b5papierrcode___ = (call) != PAPI_OK) \
    {\
        B5_ERROR("error in PAPI call: %s\ncall was %s", PAPICounters::_report_err(_b5papierrcode___), #call);\
    }\
}
#endif

//-----------------------------------------------------------------------------

namespace b5 {

struct papi_event_info
{
    int id;
    const char * name;
    const char * desc;
};

namespace {
const std::vector< papi_event_info > papi_events({
#define _b5e(e, d) {PAPI_##e, #e, d}, // utility define, undefined at the end
    _b5e(L1_DCM  , "Level 1 data cache misses")
    _b5e(L1_ICM  , "Level 1 instruction cache misses")
    _b5e(L2_DCM  , "Level 2 data cache misses")
    _b5e(L2_ICM  , "Level 2 instruction cache misses")
    _b5e(L3_DCM  , "Level 3 data cache misses")
    _b5e(L3_ICM  , "Level 3 instruction cache misses")
    _b5e(L1_TCM  , "Level 1 total cache misses")
    _b5e(L2_TCM  , "Level 2 total cache misses")
    _b5e(L3_TCM  , "Level 3 total cache misses")
    _b5e(CA_SNP  , "Snoops")
    _b5e(CA_SHR  , "Request for shared cache line (SMP)")
    _b5e(CA_CLN  , "Request for clean cache line (SMP)")
    _b5e(CA_INV  , "Request for cache line Invalidation (SMP)")
    _b5e(CA_ITV  , "Request for cache line Intervention (SMP)")
    _b5e(L3_LDM  , "Level 3 load misses")
    _b5e(L3_STM  , "Level 3 store misses")
    _b5e(BRU_IDL , "Cycles branch units are idle")
    _b5e(FXU_IDL , "Cycles integer units are idle")
    _b5e(FPU_IDL , "Cycles floating point units are idle")
    _b5e(LSU_IDL , "Cycles load/store units are idle")
    _b5e(TLB_DM  , "Data translation lookaside buffer misses")
    _b5e(TLB_IM  , "Instr translation lookaside buffer misses")
    _b5e(TLB_TL  , "Total translation lookaside buffer misses")
    _b5e(L1_LDM  , "Level 1 load misses")
    _b5e(L1_STM  , "Level 1 store misses")
    _b5e(L2_LDM  , "Level 2 load misses")
    _b5e(L2_STM  , "Level 2 store misses")
    _b5e(BTAC_M  , "BTAC miss")
    _b5e(PRF_DM  , "Prefetch data instruction caused a miss")
    _b5e(L3_DCH  , "Level 3 Data Cache Hit")
    _b5e(TLB_SD  , "TLB shootdowns (SMP)")
    _b5e(CSR_FAL , "Failed store conditional instructions")
    _b5e(CSR_SUC , "Successful store conditional instructions")
    _b5e(CSR_TOT , "Total store conditional instructions")
    _b5e(MEM_SCY , "Cycles Stalled Waiting for Memory Access")
    _b5e(MEM_RCY , "Cycles Stalled Waiting for Memory Read")
    _b5e(MEM_WCY , "Cycles Stalled Waiting for Memory Write")
    _b5e(STL_ICY , "Cycles with No Instruction Issue")
    _b5e(FUL_ICY , "Cycles with Maximum Instruction Issue")
    _b5e(STL_CCY , "Cycles with No Instruction Completion")
    _b5e(FUL_CCY , "Cycles with Maximum Instruction Completion")
    _b5e(HW_INT  , "Hardware interrupts")
    _b5e(BR_UCN  , "Unconditional branch instructions executed")
    _b5e(BR_CN   , "Conditional branch instructions executed")
    _b5e(BR_TKN  , "Conditional branch instructions taken")
    _b5e(BR_NTK  , "Conditional branch instructions not taken")
    _b5e(BR_MSP  , "Conditional branch instructions mispred")
    _b5e(BR_PRC  , "Conditional branch instructions corr. pred")
    _b5e(FMA_INS , "FMA instructions completed")
    _b5e(TOT_IIS , "Total instructions issued")
    _b5e(TOT_INS , "Total instructions executed")
    _b5e(INT_INS , "Integer instructions executed")
    _b5e(FP_INS  , "Floating point instructions executed")
    _b5e(LD_INS  , "Load instructions executed")
    _b5e(SR_INS  , "Store instructions executed")
    _b5e(BR_INS  , "Total branch instructions executed")
    _b5e(VEC_INS , "Vector/SIMD instructions executed (could include integer)")
    _b5e(RES_STL , "Cycles processor is stalled on resource")
    _b5e(FP_STAL , "Cycles any FP units are stalled")
    _b5e(TOT_CYC , "Total cycles executed")
    _b5e(LST_INS , "Total load/store inst. executed")
    _b5e(SYC_INS , "Sync. inst. executed")
    _b5e(L1_DCH  , "L1 D Cache Hit")
    _b5e(L2_DCH  , "L2 D Cache Hit")
    _b5e(L1_DCA  , "L1 D Cache Access")
    _b5e(L2_DCA  , "L2 D Cache Access")
    _b5e(L3_DCA  , "L3 D Cache Access")
    _b5e(L1_DCR  , "L1 D Cache Read")
    _b5e(L2_DCR  , "L2 D Cache Read")
    _b5e(L3_DCR  , "L3 D Cache Read")
    _b5e(L1_DCW  , "L1 D Cache Write")
    _b5e(L2_DCW  , "L2 D Cache Write")
    _b5e(L3_DCW  , "L3 D Cache Write")
    _b5e(L1_ICH  , "L1 instruction cache hits")
    _b5e(L2_ICH  , "L2 instruction cache hits")
    _b5e(L3_ICH  , "L3 instruction cache hits")
    _b5e(L1_ICA  , "L1 instruction cache accesses")
    _b5e(L2_ICA  , "L2 instruction cache accesses")
    _b5e(L3_ICA  , "L3 instruction cache accesses")
    _b5e(L1_ICR  , "L1 instruction cache reads")
    _b5e(L2_ICR  , "L2 instruction cache reads")
    _b5e(L3_ICR  , "L3 instruction cache reads")
    _b5e(L1_ICW  , "L1 instruction cache writes")
    _b5e(L2_ICW  , "L2 instruction cache writes")
    _b5e(L3_ICW  , "L3 instruction cache writes")
    _b5e(L1_TCH  , "L1 total cache hits")
    _b5e(L2_TCH  , "L2 total cache hits")
    _b5e(L3_TCH  , "L3 total cache hits")
    _b5e(L1_TCA  , "L1 total cache accesses")
    _b5e(L2_TCA  , "L2 total cache accesses")
    _b5e(L3_TCA  , "L3 total cache accesses")
    _b5e(L1_TCR  , "L1 total cache reads")
    _b5e(L2_TCR  , "L2 total cache reads")
    _b5e(L3_TCR  , "L3 total cache reads")
    _b5e(L1_TCW  , "L1 total cache writes")
    _b5e(L2_TCW  , "L2 total cache writes")
    _b5e(L3_TCW  , "L3 total cache writes")
    _b5e(FML_INS , "FM ins")
    _b5e(FAD_INS , "FA ins")
    _b5e(FDV_INS , "FD ins")
    _b5e(FSQ_INS , "FSq ins")
    _b5e(FNV_INS , "Finv ins")
    _b5e(FP_OPS  , "Floating point operations executed")
    _b5e(SP_OPS  , "Floating point operations executed; optimized to count scaled single precision vector operations")
    _b5e(DP_OPS  , "Floating point operations executed; optimized to count scaled double precision vector operations")
    _b5e(VEC_SP  , "Single precision vector/SIMD instructions")
    _b5e(VEC_DP  , "Double precision vector/SIMD instructions")
    _b5e(REF_CYC , "Reference clock cycles")
#undef _b5e
});

/// get an event by id
papi_event_info const* get_papi_event(int id)
{
    for(auto &e : papi_events)
    {
        if(e.id == id)
        {
            return &e;
        }
    }
    return nullptr;
}
} // end hidden namespace

//-----------------------------------------------------------------------------
PAPICounters::PAPICounters() : m_state(0) {}
PAPICounters::~PAPICounters()
{
    if(m_state & _STARTED)
    {
        stop();
    }
}

//-----------------------------------------------------------------------------
PAPICounters::PAPICounters(int num_events, int const* events) { init(num_events, events); }
void PAPICounters::init   (int num_events, int const* events)
{
    std::vector< int > supported;
    if(num_events == 0) // ask for all supported events
    {
        supported.reserve(papi_events.size());
        for(auto &e : papi_events)
        {
            if(PAPI_query_event(e.id) == PAPI_OK)
            {
                supported.push_back(e.id);
            }
        }
        num_events = (int)supported.size();
        events = supported.data();
    }
    m_events_asked.resize(num_events);
    int i = 0;
    for(auto &e : m_events_asked)
    {
        e.first = events[i++];
        e.second = NAN;
    }
    _init();
}

//-----------------------------------------------------------------------------
PAPICounters::PAPICounters(std::initializer_list< int > il) { init(il); }
void PAPICounters::init   (std::initializer_list< int > il)
{
    m_events_asked.resize(il.size());
    int i = 0;
    for(int ei : il)
    {
        auto &e = m_events_asked[i++];
        e.first = ei;
        e.second = NAN;
    }
    _init();
}

//-----------------------------------------------------------------------------
void PAPICounters::start()
{
    B5_ASSERT(m_state & _INIT);
    B5_CHECK_PAPI_CALL(PAPI_start_counters(m_events.data(), (int)m_events.size()));
    m_state |= _STARTED;
}

//-----------------------------------------------------------------------------
void PAPICounters::read()
{
    B5_ASSERT(m_state & _INIT);
    B5_ASSERT(m_state & _STARTED);
    B5_CHECK_PAPI_CALL(PAPI_read_counters(m_results_tmp.data(), (int)m_results_tmp.size()));
    _extract();
}

//-----------------------------------------------------------------------------
void PAPICounters::accum()
{
    // FINISH THIS
}

//-----------------------------------------------------------------------------
void PAPICounters::stop()
{
    if(m_state & _STARTED)
    {
        B5_CHECK_PAPI_CALL(PAPI_stop_counters(m_results_tmp.data(), (int)m_results_tmp.size()));
        m_state &= ~_STARTED;
    }
}

//-----------------------------------------------------------------------------
void PAPICounters::print(const char *prefix)
{
    for(auto& e : m_events_asked)
    {
        printf("%s%s%s: %lg (%s)\n",
               prefix ? prefix : "", prefix ? ": " : "",
               event_str(e.first), e.second, event_desc(e.first));
    }
}

//-----------------------------------------------------------------------------
void PAPICounters::_init()
{
    size_t num = m_events_asked.size();
    if(num == 0) return;
    m_events_pos.resize(num);

    int nc = PAPI_num_counters();
    if(nc <= PAPI_OK)
    {
        B5_PAPI_ERR(nc);
    }
    int supported = 0;
    for(auto &e : m_events_asked)
    {
        if(PAPI_query_event(e.first) == PAPI_OK)
        {
            supported++;
        }
        else
        {
            B5_WARNING("PAPI event not supported: %s (%d)\n", event_str(e.first), e.first);
        }
    }
    m_events.resize(supported);
    m_results_tmp.resize(supported);
    int i = 0, pos = 0;
    for(auto &e : m_events_asked)
    {
        if(PAPI_query_event(e.first) != PAPI_OK)
        {
            m_events_pos[i] = -1;
        }
        else
        {
            m_events[pos] = e.first;
            m_events_pos[i] = pos;
            pos++;
        }
        i++;
    }
    m_state |= _INIT;
}

//-----------------------------------------------------------------------------
void PAPICounters::_extract()
{
/*    for(int i = 0, n = (int)m_results_tmp.size(); i < n; i++)
    {
        B5_LOG("extracting: result[%d]=%lld\n", i, m_results_tmp[i]);
    }*/
    for(int i = 0, n = (int)m_events_asked.size(); i < n; i++)
    {
        int pos = m_events_pos[i];
        if(pos == -1) continue;
        m_events_asked[i].second = (double)m_results_tmp[pos];
        //B5_LOG("extracting: event_asked[%d] (%d) %s: %lld--->%lg\n", i, pos, event_to_string(m_events_asked[i].first), m_results_tmp[pos], m_events_asked[i].second);
    }
}

//-----------------------------------------------------------------------------
const char* PAPICounters::_event_str(int id, bool just_name)
{
    auto *e = get_papi_event(id);
    if(just_name)
        return e ? e->name : "(event not found)";
    else
        return e ? e->desc : "(event not found)";
}


//-----------------------------------------------------------------------------
const char* PAPICounters::_report_err(int err)
{
#define _b5e(e, d) case e: return d;

    if(err == PAPI_OK) return "";
    switch(err)
    {
    _b5e(PAPI_EINVAL     , "Invalid argument")
    _b5e(PAPI_ENOMEM     , "Insufficient memory")
    _b5e(PAPI_ESYS       , "A System/C library call failed")
    //_b5e(PAPI_ECMP       , "Not supported by component") // this is the same as PAPI_ESBSTR
    _b5e(PAPI_ESBSTR     , "Backwards compatibility")
    _b5e(PAPI_ECLOST     , "Access to the counters was lost or interrupted")
    _b5e(PAPI_EBUG       , "Internal error, please send mail to the developers")
    _b5e(PAPI_ENOEVNT    , "Event does not exist")
    _b5e(PAPI_ECNFLCT    , "Event exists, but cannot be counted due to counter resource limitations")
    _b5e(PAPI_ENOTRUN    , "EventSet is currently not running")
    _b5e(PAPI_EISRUN     , "EventSet is currently counting")
    _b5e(PAPI_ENOEVST    , "No such EventSet Available")
    _b5e(PAPI_ENOTPRESET , "Event in argument is not a valid preset")
    _b5e(PAPI_ENOCNTR    , "Hardware does not support performance counters")
    _b5e(PAPI_EMISC      , "Unknown error code")
    _b5e(PAPI_EPERM      , "Permission level does not permit operation")
    _b5e(PAPI_ENOINIT    , "PAPI hasn't been initialized yet")
    _b5e(PAPI_ENOCMP     , "Component Index isn't set")
    _b5e(PAPI_ENOSUPP    , "Not supported")
    _b5e(PAPI_ENOIMPL    , "Not implemented")
    _b5e(PAPI_EBUF       , "Buffer size exceeded")
    _b5e(PAPI_EINVAL_DOM , "EventSet domain is not supported for the operation")
    _b5e(PAPI_EATTR      , "Invalid or missing event attributes")
    _b5e(PAPI_ECOUNT     , "Too many events or attributes")
    _b5e(PAPI_ECOMBO     , "Bad combination of features")
    _b5e(PAPI_NUM_ERRORS , "Number of error messages specified in this API")
    default:
        return "unknown PAPI error";
    }

#undef _b5e
}

} // namespace b5


/*
MIT License

Copyright (c) 2017 jpmag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/