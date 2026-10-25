// Included inside worker::tests to exercise real Session/Generation/completion paths.
fn recovery_ready(sessions: &[Session], cohort: &StartupCohort) -> Vec<ReadyInput> {
    sessions.iter().map(|s| ready_input(s, cohort.allows(s))).collect()
}
fn release_recovery(cohort: &mut StartupCohort, sessions: &mut Vec<Session>,
    pool: &mut Option<PageReservations>, now: Instant, reason: &str) {
    cohort.release(sessions, pool, &mut vec![], &mut vec![], now, reason);
}

#[test]
fn recovery_mixed_visible_and_invisible_decoders_consume_once_without_holding() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
    let (old,old_rx)=session(pool.as_mut().unwrap(),128,4);
    let (silent,silent_rx)=stop_prefix_session(pool.as_mut().unwrap(),true);
    let (a,a_rx)=session(pool.as_mut().unwrap(),128,4);
    let (b,_b_rx)=session(pool.as_mut().unwrap(),512,4);
    let mut sessions=vec![old,silent,a,b];
    let old_ring=sessions[0].blocks.sliding;let silent_ring=sessions[1].blocks.sliding;
    cohort_complete(&mut sessions,old_ring,3,&mut pool);
    cohort_complete(&mut sessions,silent_ring,3,&mut pool);
    let first=sessions[0].timing.progress.first.unwrap();
    assert!(sessions[1].timing.progress.first.is_none());
    assert!(!sessions[1].generation.is_prefilling());
    assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));
    assert!(matches!(silent_rx.try_recv(),Err(mpsc::TryRecvError::Empty)));
    let mixed:Vec<_>=sessions[..3].iter().map(cohort_work).collect();
    let receipts:Vec<_>=sessions.iter().map(|s|s.timing.received).collect();
    let pages=pool.as_ref().unwrap().available_pages();
    let mut cohort=StartupCohort::new(true);
    cohort.maybe_start(&mut sessions,&mixed,Instant::now());
    assert!(cohort.active());
    assert!(sessions[..2].iter().all(|s|cohort.allows(s)&&!s.delivery.held()&&!s.delivery.once_participated()));
    assert!(sessions[2..].iter().all(|s|s.delivery.held()&&s.delivery.once_participated()));
    assert!(complete_work(&mut sessions,&mixed,Some(&[SampleOutput::Greedy(4),SampleOutput::Greedy(5),SampleOutput::Greedy(3)]),
        &mut pool,&mut vec![],&mut vec![]).unwrap());
    assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="b"));
    assert!(matches!(silent_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="ac"));
    assert!(matches!(old_rx.try_recv(),Err(mpsc::TryRecvError::Empty)));
    assert!(matches!(silent_rx.try_recv(),Err(mpsc::TryRecvError::Empty)));
    assert!(matches!(a_rx.try_recv(),Err(mpsc::TryRecvError::Empty)));
    assert_eq!(sessions[0].timing.progress.first,Some(first));
    assert_eq!(sessions[0].timing.progress.tokens,2);
    assert_eq!(sessions[1].timing.progress.tokens,2);
    assert_eq!(sessions[2].generation.completion_tokens(),1);
    assert!(sessions[2].timing.progress.first.is_none());
    assert!(sessions[2].timing.progress.deadline(DECODE_CREDIT_TARGET,DECODE_GAP_SOFTCAP).is_none());
    assert!(!cohort.allows(&sessions[2]));
    // Replay of an already consumed mixed result is rejected before any other row advances.
    assert!(complete_work(&mut sessions,&mixed,Some(&[SampleOutput::Greedy(4),SampleOutput::Greedy(5),SampleOutput::Greedy(3)]),
        &mut pool,&mut vec![],&mut vec![]).is_err());
    assert_eq!(sessions[0].generation.completion_tokens(),2);
    assert_eq!(sessions.iter().map(|s|s.timing.received).collect::<Vec<_>>(),receipts);
    assert_eq!(pool.as_ref().unwrap().available_pages(),pages);
    let available=recovery_ready(&sessions,&cohort);
    assert_eq!(available.iter().map(|r|r.count).collect::<Vec<_>>(),vec![1,1,0,128]);
    // Existing credit uses only actually emitted progress; held A is not a ready decoder.
    let mut cadence=DecodeCadence::new();
    cadence.costs[2]=Duration::from_millis(1);cadence.prompt_cost[0]=Duration::from_millis(1);
    let visible=[&sessions[0].timing.progress,&sessions[1].timing.progress];
    assert!(cadence.earned_prompt_fits(Instant::now(),&visible,1,DECODE_GAP_SOFTCAP).is_some());
}

#[test]
fn recovery_staggered_receipts_never_rehold_and_prior_checkpoint_survives_compaction() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
    let start=Instant::now();
    let (mut a,a_rx)=session(pool.as_mut().unwrap(),128,1);
    let (mut b,b_rx)=session(pool.as_mut().unwrap(),1024,4);
    let (old,old_rx)=session(pool.as_mut().unwrap(),128,2);
    a.timing.received=start;b.timing.received=start+Duration::from_millis(1000);
    let mut sessions=vec![old,a,b];
    let old_ring=sessions[0].blocks.sliding;cohort_complete(&mut sessions,old_ring,3,&mut pool);
    let first=cohort_work(&sessions[1]);let b_ring=sessions[2].blocks.sliding;
    let b_blocks=sessions[2].blocks.clone();
    let mut cohort=StartupCohort::new(true);
    cohort.maybe_start(&mut sessions,&[first],start+Duration::from_millis(2000));
    cohort_complete(&mut sessions,first.ring,3,&mut pool);
    let deadline=start+startup_cohort::RECEIPT_BUDGET;
    assert_eq!(cohort.release_reason(&sessions,deadline-Duration::from_nanos(1)),None);
    assert_eq!(cohort.release_reason(&sessions,deadline),Some("deadline"));
    release_recovery(&mut cohort,&mut sessions,&mut pool,deadline,"deadline");
    assert!(matches!(a_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));
    assert!(matches!(a_rx.try_recv(),Ok(StreamEvent::Done(_))));
    let b=sessions.iter().find(|s|s.blocks.sliding==b_ring).unwrap();
    assert!(b.delivery.once_participated()&&!b.delivery.held()&&cohort.allows(b));
    // Complete an intermediate real512 chunk, then retain the next512 checkpoint.
    let intermediate=ScheduledInput{ring:b_ring,position:0,count:512};
    assert!(complete_work(&mut sessions,&[intermediate],Some(&[SampleOutput::Logits(vec![])]),
        &mut pool,&mut vec![],&mut vec![]).unwrap());
    let checkpoint=ScheduledInput{ring:b_ring,position:512,count:512};
    let (mut c,_c_rx)=session(pool.as_mut().unwrap(),128,4);
    let (mut d,_d_rx)=session(pool.as_mut().unwrap(),128,4);
    c.timing.id=1002;d.timing.id=1003;
    c.timing.received=start+Duration::from_millis(2000);d.timing.received=start+Duration::from_millis(2100);
    assert!(!c.delivery.once_participated(),"reused ring is a new request");
    sessions.extend([c,d]);
    let now=start+Duration::from_millis(3800);
    // B is finishing but not fresh: it cannot trigger another hold of C/D.
    cohort.maybe_start(&mut sessions,&[checkpoint],now);
    assert!(!cohort.active());
    let fresh=cohort_work(sessions.iter().find(|s|s.timing.id==1002).unwrap());
    cohort.maybe_start(&mut sessions,&[fresh],now);
    assert!(cohort.active());
    assert!(sessions.iter().filter(|s|s.timing.id>=1002).all(|s|s.delivery.held()));
    let b=sessions.iter().find(|s|s.blocks.sliding==b_ring).unwrap();
    assert!(!b.delivery.held()&&cohort.allows(b));
    // An unrelated old decoder finishes, shifting Session indices under B's checkpoint.
    let b_index_before=sessions.iter().position(|s|s.blocks.sliding==b_ring).unwrap();
    assert_eq!(b_index_before,1);
    cohort_complete(&mut sessions,old_ring,4,&mut pool);
    let b_index_after=sessions.iter().position(|s|s.blocks.sliding==b_ring).unwrap();
    assert_eq!(b_index_after+1,b_index_before);
    assert_eq!(b_index_after,0);
    assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));
    assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="b"));
    assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Done(_))));
    drop(b_rx);
    let pages=pool.as_ref().unwrap().available_pages();
    assert!(!complete_work(&mut sessions,&[checkpoint],None,&mut pool,&mut vec![],&mut vec![]).unwrap());
    let b=sessions.iter().find(|s|s.blocks.sliding==b_ring).unwrap();
    assert_eq!(b.blocks.full,b_blocks.full);assert_eq!(b.generation.input().1,512);
    assert_eq!(b.timing.received,start+Duration::from_millis(1000));
    assert_eq!(pool.as_ref().unwrap().available_pages(),pages);
    let available=recovery_ready(&sessions,&cohort);let active=[b_ring];
    let selected=select_work_in(64,&available,&mut true,&mut 0,&active,true);
    assert_eq!(schedule_inputs(&available,&selected,&[checkpoint],true),vec![checkpoint]);
    assert!(complete_work(&mut sessions,&[checkpoint],Some(&[SampleOutput::Greedy(5)]),
        &mut pool,&mut vec![],&mut vec![]).unwrap());
    assert!(sessions.iter().all(|s|s.blocks.sliding!=b_ring));
    assert_eq!(pool.as_ref().unwrap().available_pages(),pages+b_blocks.full.len());
    assert!(cohort.active(),"other snapshot owns only fresh C/D");
}

#[test]
fn recovery_fresh_late_arrivals_do_not_extend_deadline_or_skip_expired_oldest() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());let start=Instant::now();
    let (mut a,_ar)=session(pool.as_mut().unwrap(),128,4);let (mut b,_br)=session(pool.as_mut().unwrap(),512,4);
    a.timing.received=start;b.timing.received=start+Duration::from_millis(100);
    let mut sessions=vec![a,b];let first=cohort_work(&sessions[0]);let mut cohort=StartupCohort::new(true);
    let partial=cohort_work(&sessions[1]);cohort.maybe_start(&mut sessions,&[partial],start+Duration::from_millis(200));
    assert!(!cohort.active());
    cohort.maybe_start(&mut sessions,&[first],start+Duration::from_millis(200));
    cohort_complete(&mut sessions,first.ring,3,&mut pool);
    let (mut late,_lr)=session(pool.as_mut().unwrap(),128,4);late.timing.received=start+Duration::from_millis(1000);
    sessions.push(late);let last=cohort_work(&sessions[2]);
    cohort.maybe_start(&mut sessions,&[last],start+Duration::from_millis(1200));
    assert!(!sessions[2].delivery.held()&&!sessions[2].delivery.once_participated()&&!cohort.allows(&sessions[2]));
    let deadline=start+startup_cohort::RECEIPT_BUDGET;
    assert_eq!(cohort.release_reason(&sessions,deadline-Duration::from_nanos(1)),None);
    assert_eq!(cohort.release_reason(&sessions,deadline),Some("deadline"));
    release_recovery(&mut cohort,&mut sessions,&mut pool,deadline,"deadline");
    assert!(sessions.iter().all(|s|cohort.allows(s)));
    let (mut younger,_yr)=session(pool.as_mut().unwrap(),128,4);younger.timing.received=start+Duration::from_millis(4700);
    sessions.push(younger);
    // Fresh late is expired at4750; do not omit it and hold only younger requests.
    let (mut youngest,_zr)=session(pool.as_mut().unwrap(),128,4);youngest.timing.received=start+Duration::from_millis(4800);
    sessions.push(youngest);let trigger=cohort_work(&sessions[3]);
    cohort.maybe_start(&mut sessions,&[trigger],start+Duration::from_millis(5000));
    assert!(!cohort.active());assert!(sessions.iter().all(|s|!s.delivery.held()));
}

#[test]
fn recovery_eos_nonstream_terminal_flush_precedes_idle_eof_and_releases_once() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
    let (mut a,ar)=session(pool.as_mut().unwrap(),128,1);a.stream=false;
    let (b,br)=session(pool.as_mut().unwrap(),128,4);
    let mut sessions=vec![a,b];let first=cohort_work(&sessions[0]);let b_ring=sessions[1].blocks.sliding;
    let mut cohort=StartupCohort::new(true);cohort.maybe_start(&mut sessions,&[first],Instant::now());
    cohort_complete(&mut sessions,first.ring,3,&mut pool);cohort_complete(&mut sessions,b_ring,crate::host::generate::EOS_TOKEN,&mut pool);
    assert!(sessions.iter().all(|s|s.delivery.terminal()&&!cohort.allows(s)));
    assert!(recovery_ready(&sessions,&cohort).iter().all(|r|r.count==0));
    assert_eq!(pool.as_ref().unwrap().available_pages(),254);
    assert!(sessions[0].timing.progress.first.is_none());
    assert_eq!(cohort.release_reason(&sessions,Instant::now()),Some("ready"));
    release_recovery(&mut cohort,&mut sessions,&mut pool,Instant::now(),"ready");
    assert!(sessions.is_empty());assert_eq!(pool.as_ref().unwrap().available_pages(),256);
    assert!(matches!(ar.try_recv(),Ok(StreamEvent::Done(o)) if o.text=="a"&&o.completion_tokens==1));
    assert!(matches!(br.try_recv(),Ok(StreamEvent::Done(o)) if o.text.is_empty()&&o.completion_tokens==0));
    assert!(matches!(ar.try_recv(),Err(mpsc::TryRecvError::Disconnected)));
    assert!(matches!(br.try_recv(),Err(mpsc::TryRecvError::Disconnected)));
    release_recovery(&mut cohort,&mut sessions,&mut pool,Instant::now(),"ready");
    assert_eq!(pool.as_ref().unwrap().available_pages(),256);
    let(tx,rx)=mpsc::channel::<()>();drop(tx);
    assert!(idle_batch::receive(&rx,64).is_none(),"worker only enters idle receive after the flush");
}

#[test]
fn recovery_silent_first_token_is_ready_but_never_gets_hidden_extra_decode() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
    let (a,ar)=stop_prefix_session(pool.as_mut().unwrap(),true);let(b,_br)=session(pool.as_mut().unwrap(),128,4);
    let mut sessions=vec![a,b];let first=cohort_work(&sessions[0]);let b_ring=sessions[1].blocks.sliding;
    let mut cohort=StartupCohort::new(true);cohort.maybe_start(&mut sessions,&[first],Instant::now());
    cohort_complete(&mut sessions,first.ring,3,&mut pool);
    assert!(!sessions[0].generation.is_prefilling()&&sessions[0].timing.progress.first.is_none());
    assert_eq!(sessions[0].generation.completion_tokens(),1);assert!(!cohort.allows(&sessions[0]));
    cohort_complete(&mut sessions,b_ring,4,&mut pool);
    assert_eq!(cohort.release_reason(&sessions,Instant::now()),Some("ready"));
    release_recovery(&mut cohort,&mut sessions,&mut pool,Instant::now(),"ready");
    assert!(matches!(ar.try_recv(),Err(mpsc::TryRecvError::Empty)));
    assert!(cohort.allows(&sessions[0]));assert!(sessions[0].delivery.once_participated());
    cohort_complete(&mut sessions,first.ring,4,&mut pool);
    assert!(matches!(ar.try_recv(),Ok(StreamEvent::Done(o)) if o.text.is_empty()));
    assert!(matches!(ar.try_recv(),Err(mpsc::TryRecvError::Disconnected)));
}

#[test]
fn recovery_cancel_and_fatal_flush_preserve_done_order_and_live_reservations() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
    let(old,old_rx)=session(pool.as_mut().unwrap(),128,4);let(a,ar)=session(pool.as_mut().unwrap(),128,1);
    let(b,br)=session(pool.as_mut().unwrap(),128,4);let(c,cr)=session(pool.as_mut().unwrap(),512,4);
    let mut sessions=vec![old,a,b,c];let old_ring=sessions[0].blocks.sliding;
    cohort_complete(&mut sessions,old_ring,3,&mut pool);
    let first=cohort_work(&sessions[1]);let b_ring=sessions[2].blocks.sliding;let c_work=cohort_work(&sessions[3]);
    let mut cohort=StartupCohort::new(true);cohort.maybe_start(&mut sessions,&[first],Instant::now());
    drop(br);drop(cr);
    cohort_complete(&mut sessions,first.ring,3,&mut pool);
    cohort_complete(&mut sessions,b_ring,crate::axes::W::SIZE,&mut pool); // Real selection error, retained.
    let pages=pool.as_ref().unwrap().available_pages();
    assert!(!complete_work(&mut sessions,&[c_work],None,&mut pool,&mut vec![],&mut vec![]).unwrap());
    assert_eq!(pool.as_ref().unwrap().available_pages(),pages);
    release_recovery(&mut cohort,&mut sessions,&mut pool,Instant::now(),"error");
    assert_eq!(sessions.len(),2); // Old decoder and unfinished C only.
    assert!(sessions.iter().all(|s|s.reservation.is_some()&&!s.delivery.terminal()));
    // The unchanged Worker fatal branch runs only after release removed held terminal owners.
    for s in &mut sessions {
        let _=s.events.send(StreamEvent::Error("offline".into()));
        release_session(s,&mut pool,&mut vec![],&mut vec![]);
    }
    sessions.clear();assert_eq!(pool.as_ref().unwrap().available_pages(),256);
    assert!(matches!(ar.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));
    assert!(matches!(ar.try_recv(),Ok(StreamEvent::Done(_))));
    assert!(matches!(ar.try_recv(),Err(mpsc::TryRecvError::Disconnected)),"no Error after Done");
    assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Delta(t)) if t=="a"));
    assert!(matches!(old_rx.try_recv(),Ok(StreamEvent::Error(t)) if t=="offline"));
    assert!(matches!(old_rx.try_recv(),Err(mpsc::TryRecvError::Disconnected)));
}

#[test]
fn recovery_no_fixed_batch_quota_and_every_capture_tracks_request_participation() {
    let mut pool=Some(PageReservations::new(64,256,CONTEXT).unwrap());
    let(mut sessions,_receivers):(Vec<_>,Vec<_>)=(0..33).map(|_|session(pool.as_mut().unwrap(),128,4)).unzip();
    let first=cohort_work(&sessions[0]);let mut cohort=StartupCohort::new(true);
    cohort.maybe_start(&mut sessions,&[first],Instant::now());
    assert!(cohort.active());assert!(sessions.iter().all(|s|s.delivery.held()&&s.delivery.once_participated()));
    let receipt=sessions[0].timing.received;let deadline=receipt+startup_cohort::RECEIPT_BUDGET;
    release_recovery(&mut cohort,&mut sessions,&mut pool,deadline,"deadline");
    assert!(sessions.iter().all(|s|cohort.allows(s)&&!s.delivery.held()&&s.delivery.once_participated()));
    cohort.maybe_start(&mut sessions,&[first],deadline);assert!(!cohort.active());
    // Legacy selection remains the explicit control, but capture also marks its request history.
    let(mut fresh,_rx):(Vec<_>,Vec<_>)=(0..2).map(|_|session(pool.as_mut().unwrap(),128,4)).unzip();
    let first=cohort_work(&fresh[0]);let mut legacy=StartupCohort::default();
    legacy.maybe_start(&mut fresh,&[first],Instant::now());
    assert!(fresh.iter().all(|s|s.delivery.once_participated()));
}
